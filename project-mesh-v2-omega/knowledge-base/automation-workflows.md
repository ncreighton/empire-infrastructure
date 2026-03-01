# Automation & Workflows

> 34 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## API Endpoints

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.6

| Endpoint | Description |
|----------|-------------|
| `GET /api/sites` | All sites with status |
| `GET /api/sites/summary` | Aggregated metrics |
| `GET /api/sites/{id}` | Single site details |
| `GET /api/workflows` | n8n workflow status |
| `GET /api/workflows/executions` | Recent executions |
| `POST /api/workflows/{id}/toggle` | Activate/deactivate |
| `GET /api/pipeline/stats` | Pipeline stage counts |
| `GET /api/pipeline/items` | Content items |
| `GET /api/analytics/summary` | Traffic summary |
| `GET /api/alerts` | Active alerts |

---

## AVOID: [X] NEVER use GPT/OpenAI for content generation

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: Claude API (Anthropic)
- **Reason**: All content uses Claude for consistency and quality

#### API Patterns

---

## AVOID: [X] NEVER use Puppeteer directly

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: Steel.dev with BrowserUse fallback
- **Reason**: Standardized on Steel.dev for session management
- **Note**: Steel.dev sessions expire after 15min idle -- implement keep-alive pings

---

## AVOID: [X] NEVER use ZimmWriter or ZimmWriter API

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: n8n content pipeline + Claude API
- **Reason**: Deprecated in favor of Claude-native workflows
- **Stage**: REMOVED

---

## Browser Automation

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 0.8

### [X] NEVER use Puppeteer directly
- **Replacement**: Steel.dev with BrowserUse fallback
- **Reason**: Standardized on Steel.dev for session management
- **Note**: Steel.dev sessions expire after 15min idle -- implement keep-alive pings

---

## Content Generation

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 1.0

### [X] NEVER use ZimmWriter or ZimmWriter API
- **Replacement**: n8n content pipeline + Claude API
- **Reason**: Deprecated in favor of Claude-native workflows
- **Stage**: REMOVED

### [X] NEVER use GPT/OpenAI for content generation
- **Replacement**: Claude API (Anthropic)
- **Reason**: All content uses Claude for consistency and quality

---

## Data Refresh Intervals

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.4

| Data | Interval | Method |
|------|----------|--------|
| Sites | 60s | htmx polling |
| Workflows | 30s | htmx polling |
| Pipeline | 60s | htmx polling |
| Analytics | 5 min | Cached |
| Alerts | 30s | htmx polling |

---

## Intelligence Systems Context

- **Source**: _empire-hub / master-context\categories\intelligence-systems.md
- **Confidence**: 1.0

- **Pattern**: FORGE+AMPLIFY pipeline (scout, enrich, expand, validate)
- **Common stack**: Python, FastAPI, SQLite knowledge codex, OpenRouter LLM
- **Projects**: Grimoire (witchcraft), VideoForge (video), VelvetVeil (printables)
- **Key principle**: Algorithmic intelligence first, LLM only for generation tasks
- **Testing**: Every system must have unit tests for all FORGE modules

---

## OPPORTUNITIES SCORED

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.4

```yaml
OPPORTUNITY_SCORE (0-10):
  Demand (0-3):
    - Based on search volume, existing sales

  Competition (0-3):
    - Lower = better score
    - Quality of competition matters

  Margin (0-2):
    - Higher prices = better

  Execution (0-2):
    - Can we make this easily?

Priority:
  8-10: HIGH - Make immediately
  6-7.9: MEDIUM - Queue for next batch
  <6: LOW - Consider later
```

---

## Substack Integration Rules

- **Source**: _empire-hub / master-context\conditionals\has-substack.md
- **Confidence**: 1.0

- Substack URL: {{project.urls.substack}}
- Rate limit: 10 API calls/minute on free tier
- Always implement exponential backoff for Substack API
- Newsletter content complements (not duplicates) site content
- Cross-promote between site and Substack
- For witchcraft vertical: Use Coven Keeper automation for engagement

---

## ZimmWriter Screens (12 navigable)

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

The screen navigator can detect and navigate to all screens via the Menu hub:
- Menu, Bulk Writer, SEO Writer, 1-Click Writer, Penny Arcade
- Local SEO Buffet, Options Menu, Advanced Triggers, Change Triggers
- AI Vault, Link Toolbox, Secret Training, Free GPTs

---

##  Workflows

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.4

### Creating a New Sabbat Kit

1. **Research Phase**
   ```
   - Gather traditional correspondences
   - Research deity associations
   - Collect traditional recipes/rituals
   - Note seasonal themes and symbols
   ```

2. **Image Generation**
   ```python
   #### Generate 3-4 images per kit:
   #### 1. Cover image (main symbol/deity)
   #### 2. Altar setup reference
   #### 3. Deity/spirit portrait (if applicable)
   #### 4. Seasonal nature scene
   
   #### Use Composio GEMINI_GENERATE_IMAGE with these settings:
   model: "gemini-3-pro-image-preview"
   image_size: "2K" or "4K"
   aspect_ratio: "3:4" for portraits, "16:9" for landscapes
   ```

3. **Content Creation**
   ```
   - Write all 12 pages of content
   - Include traditional and modern elements
   - Balance education with practical ritual
   - Add journal prompts and reflection spaces
   ```

4. **PDF Generation**
   ```bash
   python scripts/pdf_generator.py --template sabbat --sabbat [name] --output [filename].pdf
   ```

5. **Quality Check**
   - Open PDF and review all pages
   - Check image quality and placement
   - Verify fonts rendered correctly
   - Test print at home if possible

6. **Etsy Listing**
   - Create compelling title with keywords
   - Write detailed description
   - Add all relevant tags
   - Set competitive pricing ($7.99-12.99)

### Batch Generation Command
```bash
#### Generate all 8 sabbat kits
python scripts/batch_generator.py --type sabbats --all

#### Generate specific product
python scripts/pdf_generator.py --template sabbat --sabbat samhain
```

---

##  Services to Set Up

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.4

### Free / Already Have
- [OK] n8n (Contabo)
- [OK] Claude API
- [OK] Canva MCP
- [OK] Notion (free tier)
- [OK] Google Sheets
- [OK] Cloudinary (free tier: 25GB)

### Need API Keys
- Ideogram API
- Freepik API
- Runware API
- Replicate API
- Exa API

---

## General

### Active Tasks for AIDiscoveryDigest
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.8

- [ ] Streamline digest pipeline
- [ ] Grow subscriber base

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Integration Status for BulletJournals
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.8

```yaml
blog_sync: true
email_sequences: true
funnels: ['printable-templates']
automations: ['monthly-inspiration']
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

### Integration Status for CelebrationSeason
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.8

```yaml
blog_sync: true
email_sequences: true
funnels: ['party-planner']
automations: ['seasonal-tips']
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

### Integration Status for Family-Flourish
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.8

```yaml
blog_sync: true
email_sequences: true
funnels: ['family-planner']
automations: ['weekly-activity-ideas']
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

### Integration Status for WealthFromAI
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.8

```yaml
blog_sync: true
email_sequences: true
funnels: ['ai-income-blueprint']
automations: ['monetization-tips']
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

### Active Tasks for BulletJournals
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.6

- [ ] 2025 setup guides
- [ ] Printable collection

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Active Tasks for CelebrationSeason
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.6

- [ ] Winter holiday content
- [ ] New Year party guide

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Active Tasks for Family-Flourish
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.6

- [ ] Holiday family activities
- [ ] 2025 family calendar printable

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Active Tasks for ManifestAndAlign
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.6

- [ ] 2025 manifestation guide
- [ ] Vision board workshop

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Active Tasks for MythicalArchives
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.6

- [ ] Winter solstice myths series

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Active Tasks for WitchcraftForBeginners
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.6

- [ ] Yule content series
- [ ] 2025 sabbat calendar
- [ ] Beginner course launch

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### This Project Folder
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.4

```
aidiscoverydigest/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.4

```
bulletjournals/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.4

```
celebrationseason/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.4

```
family-flourish/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.4

```
manifestandalign/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.4

```
mythicalarchives/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

```
pulsegearreviews/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.4

```
smarthomewizards/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.4

```
wealthfromai/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### This Project Folder
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.4

```
witchcraftforbeginners/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---
