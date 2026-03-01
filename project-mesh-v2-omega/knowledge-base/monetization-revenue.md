# Monetization & Revenue

> 15 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## COMMANDS

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.6

```yaml
"scrape [query]":
  Execute search scrape, store results
  Example: "scrape cosmic witch sticker"

"scrape all":
  Run full keyword batch (2000+ queries)
  Takes several hours

"scrape shop [url]":
  Deep scrape a competitor shop
  Example: "scrape shop https://etsy.com/shop/MysticMoonDesigns"

"analyze":
  Run pattern detection on collected data
  Returns title patterns, price sweet spots, tag effectiveness

"analyze [niche]":
  Run analysis for specific niche
  Example: "analyze cosmic witch"

"what's selling":
  Show top bestsellers with revenue estimates

"what to make":
  Direct recommendations on products to create

"find opportunities":
  Run gap analysis, score opportunities

"find competitors":
  List top shops in niche

"patterns titles":
  Show title word/phrase patterns from bestsellers

"patterns tags":
  Show most effective tags

"patterns prices":
  Show price sweet spots by product type

"report":
  Generate full intelligence report (JSON)

"status":
  Database stats, last scrape times, data freshness
```

---

## Etsy POD Integration

- **Source**: _empire-hub / master-context\conditionals\has-etsy.md
- **Confidence**: 1.0

- Etsy shop: {{project.urls.etsy}}
- Product listings need SEO-optimized titles and tags (13 tags max)
- ForgeFiles workflow for 3D printable models (NEVER disclose AI origin)
- Seasonal collections align with site content calendar
- Product photography must be original or properly licensed

---

## Priority Queue

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.8

1. Automate all 16 WordPress sites for hands-off content + design
2. Scale KDP book publishing operations
3. Launch AI Lead Magnet Generator business
4. Launch Newsletter-as-a-Service business
5. Expand Etsy POD empire (cosmic, cottage, green, sea witch sub-niches)
6. Transition all content generation to Claude/n8n (away from ZimmWriter)

---

## Your Role

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.6

- Deploy, configure, and maintain the OpenClaw gateway on Contabo server
- Manage all 16 WordPress sites via REST API and WP-CLI
- Generate, schedule, and publish content matching each site's brand voice
- Trigger and manage n8n automation workflows
- Control Android phone hardware via paired Termux node
- Track revenue, manage KDP books, and run Etsy POD operations
- Execute boldly. Ship fast. Automate everything.

---

## revenue-streams

### Revenue streams: digital-products
- **Source**: 3d-print-forge / manifest.json
- **Confidence**: 1.0

Revenue streams: digital-products

---

### Revenue streams: affiliate, digital-products
- **Source**: ai-discovery-digest / manifest.json
- **Confidence**: 1.0

Revenue streams: affiliate, digital-products

---

### Revenue streams: memberships, tips
- **Source**: bmc-witchcraft / manifest.json
- **Confidence**: 1.0

Revenue streams: memberships, tips

---

### Revenue streams: affiliate, digital-products, etsy-pod
- **Source**: bullet-journals / manifest.json
- **Confidence**: 1.0

Revenue streams: affiliate, digital-products, etsy-pod

---

### Revenue streams: affiliate
- **Source**: celebration-season / manifest.json
- **Confidence**: 1.0

Revenue streams: affiliate

---

### Revenue streams: digital-products, etsy-pod
- **Source**: etsy-agent-v2 / manifest.json
- **Confidence**: 1.0

Revenue streams: digital-products, etsy-pod

---

### Revenue streams: etsy-pod
- **Source**: pod-automation-system / manifest.json
- **Confidence**: 1.0

Revenue streams: etsy-pod

---

### Revenue streams: affiliate, digital-products, substack, etsy-pod
- **Source**: witchcraft-for-beginners / manifest.json
- **Confidence**: 1.0

Revenue streams: affiliate, digital-products, substack, etsy-pod

---

##  Project Overview

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.6

VelvetVeilPrintables is an Etsy digital download business specializing in premium witchcraft, spirituality, and mystical printables. This Claude Code project automates the creation of high-quality PDF ritual kits, journals, planners, and other digital products.

**Business Model**: Digital downloads on Etsy → passive income at scale
**Target Market**: Modern witches, spiritual practitioners, pagan community
**Quality Standard**: Premium products that justify $5-15 price points

---

## General

### Integration Status for MythicalArchives
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: true
email_sequences: true
funnels: ['mythology-course']
automations: ['myth-of-the-week']
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

### Integration Status for WitchcraftForBeginners
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: true
email_sequences: true
funnels: ['lead-magnet', 'course-sales']
automations: ['welcome-sequence', 'sabbat-reminders']
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
