# SEO & Search Optimization

> 59 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## AVOID: [X] NEVER use Yoast SEO plugin

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: RankMath
- **Reason**: Standardized across all sites on RankMath

---

## Core Modules to Build

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 0.8

1. **Article Generator** - Multi-stage content pipeline with quality gates
2. **SEO Optimizer** - SERP analysis, NLP keywords, content scoring
3. **Image Generator** - AI images contextual to content
4. **WordPress Publisher** - Multi-site publishing with scheduling
5. **Humanizer Engine** - AI detection bypass
6. **Internal Linker** - Smart linking across site content

---

## Environment Variables Required

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 0.4

```bash
#### AI APIs
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

#### Database
DATABASE_URL=postgresql://user:pass@localhost:5432/nsce

#### WordPress (per site)
WFB_WP_URL=https://witchcraftforbeginners.com/wp-json/wp/v2
WFB_WP_USER=
WFB_WP_PASSWORD=
#### ... repeat for all 16 sites

#### SEO APIs
SERP_API_KEY=
DATAFORSEO_KEY=

#### Quality Checking
ORIGINALITY_AI_KEY=
```

---

## Image Hosting Rules (CRITICAL)

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.6

**For any images displayed on WordPress sites:**

1. **NEVER use ngrok URLs** - they are temporary and will break
2. **ALWAYS upload to WordPress** media library for content images
3. **SEO-optimized filenames required:**
   - Format: `{keyword}-{descriptor}-{site}.png`
   - Example: `smart-home-guide-featured-smarthomewizards.png`
4. **ALWAYS set alt text** with target keyword
5. **ALWAYS set title attribute** for accessibility

**For Pinterest pins (exception):**
- Cloudinary CDN is acceptable for pin images since Pinterest hosts the final image
- However, if the pin image is also used as featured image on WordPress, upload to WordPress first

```python
#### For content/featured images - ALWAYS WordPress
wp_url = upload_to_wordpress(site_id, image_path, seo_filename)

#### WRONG - Never for WordPress content
image_url = "https://something.ngrok.dev/image.png"  # NEVER
```

---

## Important Notes

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 0.4

- Always maintain voice consistency per site
- Run AI detection check before publishing
- Use internal linking for all articles (3-5 links minimum)
- Generate meta title and description for every article
- Include FAQ schema where appropriate

---

## Key API Endpoints

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

### Connection & Status
- `POST /connect` -- Connect to running ZimmWriter
- `POST /launch` -- Launch ZimmWriter and connect
- `GET /status` -- Full status dump (all controls, states)
- `GET /is-running` -- Check if ZimmWriter process exists

### Bulk Writer Controls
- `POST /configure` -- Set all dropdown settings
- `POST /checkboxes` -- Set all checkbox states
- `POST /feature-toggle` -- Toggle right-side feature buttons
- `POST /titles` -- Set bulk article titles
- `POST /load-csv` -- Load SEO CSV file
- `POST /start` -- Start Bulk Writer
- `POST /stop` -- Stop Bulk Writer

### Config Windows (11 feature toggles)
- `POST /config/wordpress` -- WordPress upload settings
- `POST /config/serp-scraping` -- SERP scraping settings
- `POST /config/deep-research` -- Deep Research model + link counts
- `POST /config/link-pack` -- Link pack selection
- `POST /config/style-mimic` -- Style mimic text
- `POST /config/custom-outline` -- Custom outline template
- `POST /config/custom-prompt` -- Custom editorial prompt
- `POST /config/youtube-videos` -- YouTube video embedding
- `POST /config/webhook` -- Webhook URL
- `POST /config/alt-images` -- Alt image model selection
- `POST /config/seo-csv` -- SEO CSV file path

### Campaign Intelligence
- `POST /campaign/plan` -- Plan campaign (classify titles, select settings)
- `POST /campaign/run` -- Plan + execute campaign on one site
- `POST /campaign/batch` -- Run campaigns across multiple sites
- `POST /campaign/classify` -- Classify titles without planning

### Screen Navigation
- `GET /screen/current` -- Detect current ZimmWriter screen
- `GET /screen/available` -- List all navigable screens
- `POST /screen/navigate` -- Navigate to any screen
- `POST /screen/menu` -- Return to Menu hub

### Link Packs
- `POST /link-packs/build` -- Build link pack for one site
- `POST /link-packs/build-all` -- Build link packs for all sites
- `GET /link-packs/list` -- List saved link pack files

### Presets & Orchestration
- `GET /presets` -- List all domain presets
- `POST /presets/{domain}/apply` -- Apply site preset
- `POST /orchestrate` -- Multi-site sequential job runner
- `POST /run-job` -- Complete end-to-end single job

---

## SEO Standards

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 1.0

- Target featured snippets with structured H2/H3 content
- E-E-A-T signals in every article
- RankMath optimization: focus keyword in first paragraph, meta description, FAQ schema
- Schema markup: BlogPosting (default), HowTo, FAQPage, Product as appropriate
- Internal linking within content clusters for topical authority

---

## Tech Stack

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.6

- **Runtime**: Node.js 22+
- **Gateway**: OpenClaw (`npm install -g openclaw@latest`)
- **Model**: `anthropic/claude-opus-4-5` (primary), `anthropic/claude-sonnet-4-5` (fallback)
- **Themes**: Blocksy (15 sites) + Astra (Family-Flourish)
- **SEO**: RankMath Pro (NOT Yoast)
- **Cache**: LiteSpeed Cache (NOT WP Rocket)
- **MCP**: AI Engine plugin
- **Security**: Wordfence | **Backups**: UpdraftPlus | **Affiliate**: Content Egg
- **Snippets**: WPCode | **GDPR**: Complianz | **TOC**: Easy Table of Contents

---

## WordPress

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 0.6

### [X] NEVER use Yoast SEO plugin
- **Replacement**: RankMath
- **Reason**: Standardized across all sites on RankMath

### [X] NEVER edit theme files directly
- **Replacement**: Use child theme or Blocksy customizer
- **Reason**: Updates will overwrite direct edits

---

##  DESIGN SYSTEM

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.4

### Brand Identity

```
BRAND NAME: The Connected Haven
TAGLINE: "Smart Home Technology Simplified"
BRAND ESSENCE: Sanctuary meets technology - your home as an intelligent, welcoming space

BRAND PERSONALITY:
- Approachable Expert (not intimidating techie)
- Trusted Neighbor (not corporate reviewer)
- Problem Solver (not feature lister)
- Home-First Thinker (not gadget collector)
```

### Color Palette

```css
:root {
  /* Primary - Haven Blue (trust, technology, calm) */
  --haven-primary: #2563eb;
  --haven-primary-dark: #1d4ed8;
  --haven-primary-light: #60a5fa;
  
  /* Secondary - Warm Home (comfort, approachability) */
  --haven-secondary: #f59e0b;
  --haven-secondary-dark: #d97706;
  --haven-secondary-light: #fbbf24;
  
  /* Accent - Connected Green (smart, active, eco) */
  --haven-accent: #10b981;
  --haven-accent-dark: #059669;
  --haven-accent-light: #34d399;
  
  /* Neutrals */
  --haven-dark: #0f172a;
  --haven-text: #1e293b;
  --haven-muted: #64748b;
  --haven-light: #f1f5f9;
  --haven-white: #ffffff;
  
  /* Ecosystem Colors */
  --alexa-blue: #00caff;
  --homekit-orange: #ff6b00;
  --google-blue: #4285f4;
  --smartthings-green: #15bfab;
  --matter-purple: #7c3aed;
  
  /* Status Colors */
  --connected-green: #22c55e;
  --warning-amber: #f59e0b;
  --offline-red: #ef4444;
  
  /* Gradients */
  --haven-gradient: linear-gradient(135deg, var(--haven-primary) 0%, var(--haven-accent) 100%);
  --hero-gradient: linear-gradient(180deg, rgba(15,23,42,0.9) 0%, rgba(15,23,42,0.7) 100%);
}
```

### Typography System

```css
/* Font Stack - NO Roboto/Inter (per Anti-Patterns) */
:root {
  /* Headers - Modern geometric sans */
  --font-display: 'Plus Jakarta Sans', 'DM Sans', system-ui, sans-serif;
  
  /* Body - Highly readable */
  --font-body: 'Source Sans 3', 'Open Sans', system-ui, sans-serif;
  
  /* Mono - Code/specs */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  
  /* Scale */
  --text-xs: clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem);
  --text-sm: clamp(0.875rem, 0.8rem + 0.375vw, 1rem);
  --text-base: clamp(1rem, 0.9rem + 0.5vw, 1.125rem);
  --text-lg: clamp(1.125rem, 1rem + 0.625vw, 1.25rem);
  --text-xl: clamp(1.25rem, 1.1rem + 0.75vw, 1.5rem);
  --text-2xl: clamp(1.5rem, 1.25rem + 1.25vw, 2rem);
  --text-3xl: clamp(1.875rem, 1.5rem + 1.875vw, 2.5rem);
  --text-4xl: clamp(2.25rem, 1.75rem + 2.5vw, 3.5rem);
  --text-5xl: clamp(3rem, 2rem + 5vw, 5rem);
}

/* Typography Styles */
h1, .h1 {
  font-family: var(--font-display);
  font-size: var(--text-4xl);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.02em;
}

h2, .h2 {
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.01em;
}

h3, .h3 {
  font-family: var(--font-display);
  font-size: var(--text-2xl);
  font-weight: 600;
  line-height: 1.3;
}

body, p {
  font-family: var(--font-body);
  font-size: var(--text-base);
  font-weight: 400;
  line-height: 1.7;
}
```

### Spacing System

```css
:root {
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.5rem;    /* 24px */
  --space-6: 2rem;      /* 32px */
  --space-8: 3rem;      /* 48px */
  --space-10: 4rem;     /* 64px */
  --space-12: 6rem;     /* 96px */
  --space-16: 8rem;     /* 128px */
  
  --container-sm: 640px;
  --container-md: 768px;
  --container-lg: 1024px;
  --container-xl: 1280px;
  --container-2xl: 1536px;
}
```

---

##  EXECUTIVE SUMMARY

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

### Current State Analysis
- **Theme:** Blocksy (wp-theme-blocksy)
- **Page Builder:** Elementor 3.33.4
- **SEO:** RankMath (standard, NOT Pro)
- **MegaMenu:** MegaMenu Pro active
- **Template:** Custom template-homepage.php
- **GDPR:** Complianz
- **Analytics:** Site Kit by Google 1.167.0
- **WordPress:** 6.9

### Critical Issues Identified
1. **SEO:** Generic meta description ("Discover the best smart home devices today." = weak)
2. **Branding:** Using Unsplash stock photo as OG image (not branded)
3. **Author:** Shows email (creightonnick0@gmail.com) - needs persona
4. **RankMath:** Standard version (missing Pro schema features)
5. **Design:** Most basic Blocksy implementation of all 4 sites

### Unique Position
- Focused on PRODUCT REVIEWS (vs TheConnectedHaven's guides/ecosystem focus)
- Review-centric = affiliate revenue focused
- Complementary to TheConnectedHaven (can cross-link)

### Transformation Goals
- Differentiate clearly from TheConnectedHaven (reviews vs guides)
- Build comprehensive product review infrastructure
- Maximize affiliate conversion through trust signals
- Create "Lab Tested" brand positioning
- Establish review authority with rigorous testing framework

---

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.6

### Current State Analysis
- **Theme:** Blocksy (confirmed: wp-theme-blocksy, ct- class prefixes)
- **Page Builder:** Elementor 3.33.3
- **SEO:** RankMath Pro (rank-math-schema-pro)
- **Caching:** LiteSpeed Cache
- **GDPR:** Complianz
- **Analytics:** Site Kit by Google 1.167.0
- **WordPress:** 6.9

### Critical Issues Identified
1. **SEO:** Meta description too short ("Smart Home Guide" = 16 chars vs 120-160 recommended)
2. **Branding:** Generic title structure, missing OG image
3. **Author Display:** Shows email (creightonnick0@gmail.com) instead of brand persona
4. **Navigation:** 25+ pages in flat structure need hierarchical mega menu
5. **Design:** Default Blocksy implementation lacks distinctive "Modern Tech Picasso" aesthetic

### Transformation Goals
- Convert from generic smart home site to THE definitive smart home authority
- Implement distinctive visual identity that stands out from CNET/Tom's Guide
- Create immersive ecosystem experience for each platform (Alexa, HomeKit, Google, SmartThings)
- Maximize lead capture with strategic free resource delivery
- Build topical authority through content architecture redesign

---

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.4

### Current State Analysis
- **Theme:** Blocksy + Child Theme (wp-theme-blocksy, blocksy-child)
- **Page Builder:** Elementor 3.33.4
- **SEO:** RankMath Pro (rank-math-schema-pro)
- **Caching:** LiteSpeed Cache
- **Dark Mode:** Implemented (clearainews-dark class)
- **MegaMenu:** MegaMenu Pro active
- **Translation:** TranslatePress (translatepress-en_US)
- **GDPR:** Complianz
- **Analytics:** Site Kit by Google 1.167.0
- **WordPress:** 6.9
- **Logo:** Custom uploaded (Clear-AI-News-Logo.jpeg)

### Unique Strengths
1. **Author Persona Established:** "Alex Clearfield" already configured
2. **Dark Mode Active:** Modern aesthetic in place
3. **Strong Meta:** "Where artificial intelligence meets human understanding. We decode the AI revolution so you don't have to."
4. **Pro Plugins:** Full Blocksy Companion Pro suite active

### Transformation Goals
- Evolve from basic AI news site to THE destination for accessible AI journalism
- Differentiate from MIT Tech Review (academic) and VentureBeat (business-focused)
- Create "human translator" brand positioning
- Build interactive learning elements alongside news
- Establish thought leadership through distinctive visual journalism

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

### Current State Analysis
- **Theme:** Custom (wp-theme-wgr-design-system) - Already distinctive!
- **Page Builder:** Elementor 3.33.4
- **SEO:** RankMath Pro (rank-math-schema-pro)
- **Affiliate:** Content Egg (ecs_ajax_params detected)
- **GDPR:** Complianz
- **Analytics:** Site Kit by Google 1.167.0
- **WordPress:** 6.9

### Existing Design System Strengths
```
DETECTED CUSTOM CLASSES:
- wgr-hero (hero section)
- wgr-header (header component)
- wgr-logo (logo styling)
- wgr-nav (navigation)
- wgr-design-system (theme identifier)
- wgr-home (homepage body class)

LOGO STRUCTURE:
"Wearable" (primary) + "Gear" (accent) + "Reviews" (secondary)
Tagline: "Your Daily Pulse on Wearable Tech"
Icon: ⌚ (watch emoji as pulse indicator)
```

### Current Meta (Strong!)
```
Title: Home - Wearable Gear Reviews
Description: "Data-driven reviews that cut through the hype. Make smarter choices with wearable technology that truly fits your lifestyle."
```

### Transformation Goals
- Enhance existing custom design system (don't replace)
- Build comprehensive comparison and buying guide infrastructure
- Create interactive fitness/health tracking comparisons
- Maximize affiliate revenue through strategic product placement
- Establish authority in fitness wearables niche

---

## ️ SITE ARCHITECTURE

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.4

### Navigation Structure (Mega Menu)

```
PRIMARY NAVIGATION:
├── Get Started
│   ├── Smart Home 101 (pillar page)
│   ├── Beginner's Roadmap
│   ├── Budget Planning Calculator
│   └── Ecosystem Quiz
│
├── Ecosystems ▼ (mega menu with icons)
│   ├──  Amazon Alexa
│   │   ├── Alexa Hub Guide
│   │   ├── Best Alexa Devices
│   │   ├── Routines & Automation
│   │   └── Commands Cheat Sheet
│   ├──  Apple HomeKit
│   │   ├── HomeKit Hub Guide
│   │   ├── Best HomeKit Devices
│   │   ├── Scenes & Automation
│   │   └── Shortcuts Library
│   ├──  Google Home
│   │   ├── Google Hub Guide
│   │   ├── Best Google Devices
│   │   ├── Routines Setup
│   │   └── Commands Reference
│   └──  SmartThings
│       ├── SmartThings Hub Guide
│       ├── Compatible Devices
│       ├── Automation Blueprints
│       └── Edge Drivers Guide
│
├── Guides ▼
│   ├── By Room
│   │   ├── Living Room
│   │   ├── Kitchen
│   │   ├── Bedroom
│   │   ├── Bathroom
│   │   └── Outdoor
│   ├── By Device Type
│   │   ├── Smart Speakers
│   │   ├── Smart Displays
│   │   ├── Smart Lighting
│   │   ├── Smart Thermostats
│   │   ├── Smart Locks
│   │   └── Smart Cameras
│   └── By Goal
│       ├── Energy Savings
│       ├── Security
│       ├── Entertainment
│       └── Accessibility
│
├── Tools
│   ├── Ecosystem Quiz
│   ├── ROI Calculator
│   ├── Budget Planner
│   ├── Compatibility Checker
│   └── Troubleshooting Guide
│
├── Blog
│
└── About
    ├── About Us
    ├── Editorial Policy
    └── Contact
```

### Content Hub Structure

```
PILLAR PAGES (10,000+ words each):
1. /smart-home-beginners-guide/ - Ultimate beginner resource
2. /alexa-ecosystem/ - Complete Alexa guide
3. /apple-homekit-ecosystem/ - Complete HomeKit guide
4. /google-home-ecosystem/ - Complete Google guide
5. /smartthings-ecosystem/ - Complete SmartThings guide
6. /matter-thread-guide/ - Future-proofing guide
7. /smart-home-security-guide/ - Security deep dive
8. /energy-savings-automation/ - Cost savings guide

CLUSTER CONTENT (supporting articles):
- Link each cluster article to parent pillar
- 20-30 supporting articles per pillar
- Internal linking matrix
```

---

##  SUCCESS METRICS

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 1.0

### KPIs to Track

```
TRAFFIC:
- Organic sessions (target: +50% in 6 months)
- Pages per session (target: 3+)
- Average session duration (target: 3+ minutes)
- Bounce rate (target: <50%)

ENGAGEMENT:
- Quiz completions (target: 500/month)
- Calculator usage (target: 300/month)
- Resource downloads (target: 1000/month)
- Comments per post (target: 5+)

CONVERSION:
- Email signups (target: 500/month)
- Lead magnet downloads
- Affiliate click-through rate
- Newsletter open rate (target: 30%+)

SEO:
- Keyword rankings (track top 50 targets)
- Featured snippets captured
- Domain authority growth
- Backlink acquisition
```

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

```
TRAFFIC:
- 50,000 monthly sessions in 6 months
- 2+ pages per session
- 5+ minute avg session (review depth)
- <50% bounce rate

REVENUE:
- Affiliate CTR: 10%+ on reviews
- Conversion rate: 4%+ on affiliate clicks
- Monthly affiliate revenue: $3,000+
- "Where to Buy" engagement: 25%+

ENGAGEMENT:
- Review completion rate: 40%+
- FAQ expansion rate: 30%+
- Comparison table views: 50%+
- Newsletter signups: 200/month

SEO:
- Review rich results in SERPs
- Featured snippets for "best X" queries
- Top 10 for 30+ product reviews
- Product panel appearances
```

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

```
TRAFFIC:
- 75,000 monthly sessions in 6 months
- 2.5+ pages per session
- 4+ minute avg session (review depth)
- <45% bounce rate

REVENUE:
- Affiliate click rate: 8%+ on reviews
- Conversion rate: 3%+ on affiliate links
- Average order value tracking
- Monthly affiliate revenue: $2,000+

ENGAGEMENT:
- Comparison tool usage: 500/month
- Newsletter signups: 300/month
- Social shares per review: 25+
- Return visitor rate: 25%+

SEO:
- Featured snippets for "best X" queries
- Top 10 for 50+ product reviews
- Review rich results in SERPs
- Product carousel inclusion
```

---

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.4

```
TRAFFIC GOALS:
- 50,000 monthly sessions within 6 months
- 3+ pages per session
- 4+ minute average session duration
- <40% bounce rate

ENGAGEMENT:
- Newsletter signups: 1,000/month
- Social shares per article: 50+
- Comments per article: 10+
- Return visitor rate: 30%+

SEO:
- Featured snippets for AI definitions
- Top 10 rankings for "AI news" queries
- Google News inclusion
- Discover traffic
```

---

##  CONTENT TEMPLATES

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.6

### Blog Post Template Structure

```
1. Hero Image (16:9, custom branded)
2. Title (H1)
3. Meta info (date, author, read time, category)
4. Introduction (hook + what you'll learn)
5. Quick Answer Box (featured snippet target)
6. Table of Contents (auto-generated)
7. Main Content Sections (H2s)
8. Product Recommendations (if applicable)
9. FAQ Section (schema-enhanced)
10. Key Takeaways Box
11. Author Bio
12. Related Posts
13. Comments
14. Newsletter Signup
```

### Comparison Page Template

```
1. Title: "[Product A] vs [Product B]: Which Is Right for You?"
2. Quick Verdict Box
3. Comparison Table (sticky header)
4. Detailed Comparison Sections:
   - Design & Build
   - Features
   - Performance
   - Ecosystem Compatibility
   - Price & Value
5. Who Should Buy What
6. FAQ
7. Final Recommendation
```

---

**END OF BLUEPRINT**

*This document serves as the complete transformation guide for TheConnectedHaven.com. Execute phases sequentially, validate each change, and iterate based on analytics data.*

**Document Version:** 1.0
**Created:** 2025-12-16
**Author:** Claude (AI Publishing Empire Assistant)

---

##  SEO & META OPTIMIZATION

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.6

### Homepage Meta

```php
// Already strong - minor enhancements
Title: Clear AI News | AI News Explained for Humans
Meta Description: Where artificial intelligence meets human understanding. Daily AI news, analysis, and explainers that cut through the hype. No PhD required.
Canonical: https://clearainews.com/

// Open Graph (use existing logo)
og:image: https://clearainews.com/wp-content/uploads/2025/11/Clear-AI-News-Logo.jpeg
og:image:width: 1200
og:image:height: 630
```

### Schema Enhancement

```json
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "NewsMediaOrganization",
      "@id": "https://clearainews.com/#organization",
      "name": "Clear AI News",
      "url": "https://clearainews.com",
      "logo": {
        "@type": "ImageObject",
        "url": "https://clearainews.com/wp-content/uploads/2025/11/Clear-AI-News-Logo.jpeg"
      },
      "description": "AI news and analysis for everyone",
      "foundingDate": "2025",
      "founder": {
        "@type": "Person",
        "name": "Alex Clearfield"
      },
      "publishingPrinciples": "https://clearainews.com/editorial-standards/",
      "diversityStaffingReport": "https://clearainews.com/about/"
    },
    {
      "@type": "Person",
      "@id": "https://clearainews.com/#alex-clearfield",
      "name": "Alex Clearfield",
      "url": "https://clearainews.com/author/clearaieditor/",
      "jobTitle": "AI Correspondent & Editor-in-Chief",
      "worksFor": {"@id": "https://clearainews.com/#organization"},
      "description": "Making AI news accessible to everyone"
    }
  ]
}
```

---

##  SEO IMPLEMENTATION

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 1.0

### Meta Tags & Schema

```php
// RankMath Settings - Homepage
Title: The Connected Haven | Smart Home Technology Simplified
Meta Description: Transform your home into an intelligent sanctuary. Expert guides, hands-on reviews, and smart home automation tutorials for Alexa, HomeKit, Google Home & SmartThings.
Canonical: https://theconnectedhaven.com/

// Open Graph
og:title: The Connected Haven - Smart Home Made Simple
og:description: Your trusted guide to smart home technology. From beginner basics to advanced automation, we help you build the connected home of your dreams.
og:image: [Custom branded OG image - 1200x630px]
og:type: website

// Twitter Card
twitter:card: summary_large_image
twitter:site: @ConnectedHaven
twitter:creator: @ConnectedHaven
```

### Schema Markup (JSON-LD)

```json
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://theconnectedhaven.com/#organization",
      "name": "The Connected Haven",
      "url": "https://theconnectedhaven.com",
      "logo": {
        "@type": "ImageObject",
        "url": "https://theconnectedhaven.com/wp-content/uploads/logo.png",
        "width": 512,
        "height": 512
      },
      "description": "Expert smart home technology guides and tutorials",
      "sameAs": [
        "https://twitter.com/ConnectedHaven",
        "https://youtube.com/@ConnectedHaven",
        "https://pinterest.com/connectedhaven"
      ]
    },
    {
      "@type": "WebSite",
      "@id": "https://theconnectedhaven.com/#website",
      "url": "https://theconnectedhaven.com",
      "name": "The Connected Haven",
      "publisher": {"@id": "https://theconnectedhaven.com/#organization"},
      "potentialAction": {
        "@type": "SearchAction",
        "target": "https://theconnectedhaven.com/?s={search_term_string}",
        "query-input": "required name=search_term_string"
      }
    }
  ]
}
```

### Author Persona Setup

```
AUTHOR PROFILE: "Haven Team" or Create Persona
Name: Jordan Haven (or "The Connected Haven Team")
Role: Smart Home Editor
Bio: Smart home enthusiast who's tested 500+ devices across every major ecosystem. 
     On a mission to help families simplify their connected homes.
Avatar: Custom illustrated avatar (not stock photo)
Author URL: /author/jordan-haven/
```

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

### Meta Optimization

```php
// CRITICAL: Fix weak homepage meta
Title: Smart Home Gear Reviews | Lab-Tested Reviews You Can Trust
Meta Description: Expert smart home product reviews with rigorous 30-day testing. Unbiased ratings for smart lights, security cameras, thermostats, speakers & more. Find your perfect device.

// Open Graph (REPLACE UNSPLASH IMAGE)
og:image: [Custom branded OG image - 1200x630px]
og:title: Smart Home Gear Reviews - Lab-Tested Reviews
og:description: Expert smart home product reviews with rigorous testing. Find your perfect smart home devices.
```

### Review Schema (RankMath Configuration)

```json
{
  "@context": "https://schema.org",
  "@type": "Review",
  "itemReviewed": {
    "@type": "Product",
    "name": "Philips Hue White & Color Ambiance Starter Kit",
    "brand": {
      "@type": "Brand",
      "name": "Philips Hue"
    },
    "image": "product-image.jpg",
    "sku": "563296",
    "gtin13": "0046677563295",
    "offers": {
      "@type": "Offer",
      "price": "179.99",
      "priceCurrency": "USD",
      "availability": "https://schema.org/InStock",
      "url": "affiliate-link"
    }
  },
  "reviewRating": {
    "@type": "Rating",
    "ratingValue": "9.1",
    "bestRating": "10",
    "worstRating": "1"
  },
  "author": {
    "@type": "Organization",
    "name": "Smart Home Gear Reviews",
    "url": "https://smarthomegearreviews.com"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Smart Home Gear Reviews"
  },
  "reviewBody": "Full review content...",
  "positiveNotes": {
    "@type": "ItemList",
    "itemListElement": [
      {"@type": "ListItem", "position": 1, "name": "Excellent color accuracy"}
    ]
  },
  "negativeNotes": {
    "@type": "ItemList",
    "itemListElement": [
      {"@type": "ListItem", "position": 1, "name": "Requires hub"}
    ]
  }
}
```

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

### Review Schema

```json
{
  "@context": "https://schema.org",
  "@type": "Review",
  "itemReviewed": {
    "@type": "Product",
    "name": "Apple Watch Series 10",
    "brand": {
      "@type": "Brand",
      "name": "Apple"
    },
    "image": "product-image.jpg",
    "offers": {
      "@type": "AggregateOffer",
      "lowPrice": "399",
      "highPrice": "799",
      "priceCurrency": "USD",
      "offerCount": "4"
    }
  },
  "reviewRating": {
    "@type": "Rating",
    "ratingValue": "9.2",
    "bestRating": "10",
    "worstRating": "1"
  },
  "author": {
    "@type": "Organization",
    "name": "Wearable Gear Reviews"
  },
  "publisher": {
    "@type": "Organization",
    "name": "Wearable Gear Reviews",
    "logo": {
      "@type": "ImageObject",
      "url": "https://wearablegearreviews.com/logo.png"
    }
  },
  "reviewBody": "Full review content...",
  "positiveNotes": {
    "@type": "ItemList",
    "itemListElement": [
      {"@type": "ListItem", "position": 1, "name": "Excellent display"},
      {"@type": "ListItem", "position": 2, "name": "Great fitness tracking"}
    ]
  },
  "negativeNotes": {
    "@type": "ItemList",
    "itemListElement": [
      {"@type": "ListItem", "position": 1, "name": "Battery life could be better"}
    ]
  }
}
```

### FAQ Schema (for buying guides)

```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What's the best smartwatch for fitness?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Based on our testing, the Apple Watch Series 10 offers the best overall fitness tracking..."
      }
    }
  ]
}
```

---

##  ARTICLE PAGE TEMPLATE

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.4

### Single Post Layout

```html
<!-- Article Page Structure -->
<article class="clear-article" itemscope itemtype="https://schema.org/NewsArticle">
  <!-- Article Header -->
  <header class="article-header">
    <div class="container container--narrow">
      <div class="article-meta">
        <a href="/category/analysis/" class="article-category cat-analysis">
          Analysis
        </a>
        <time datetime="2025-12-16" itemprop="datePublished">
          December 16, 2025
        </time>
        <span class="article-readtime">8 min read</span>
      </div>
      
      <h1 class="article-title" itemprop="headline">
        OpenAI's o3 Changes Everything--Here's What It Means for You
      </h1>
      
      <p class="article-subtitle" itemprop="description">
        The new reasoning model isn't just another GPT upgrade. 
        We break down why this release actually matters.
      </p>
      
      <div class="article-author" itemprop="author" itemscope itemtype="https://schema.org/Person">
        <img src="/author-alex.jpg" alt="" class="author-avatar">
        <div class="author-info">
          <a href="/author/clearaieditor/" class="author-name" itemprop="name">
            Alex Clearfield
          </a>
          <span class="author-role">AI Correspondent</span>
        </div>
        <div class="article-share">
          <button class="share-btn" data-share="twitter">𝕏</button>
          <button class="share-btn" data-share="linkedin">in</button>
          <button class="share-btn" data-share="copy"></button>
        </div>
      </div>
    </div>
  </header>
  
  <!-- Featured Image -->
  <figure class="article-hero">
    <img src="..." alt="" itemprop="image">
    <figcaption>Image credit / AI-generated illustration</figcaption>
  </figure>
  
  <!-- Article Body -->
  <div class="article-content" itemprop="articleBody">
    <div class="container container--narrow">
      
      <!-- Key Takeaways Box -->
      <aside class="key-takeaways">
        <h4>Key Takeaways</h4>
        <ul>
          <li>Point one explained simply</li>
          <li>Point two explained simply</li>
          <li>Point three explained simply</li>
        </ul>
      </aside>
      
      <!-- Content -->
      <p class="lede">Opening paragraph with hook...</p>
      
      <h2>Section Heading</h2>
      <p>Content...</p>
      
      <!-- AI Term Tooltip -->
      <p>
        The model uses <span class="ai-term" data-definition="Large Language Model - an AI trained on vast text data">LLM</span> 
        architecture with improved...
      </p>
      
      <!-- Pull Quote -->
      <blockquote class="pull-quote">
        "Significant quote from the article that captures attention."
      </blockquote>
      
      <!-- Code/Technical Block -->
      <div class="code-block">
        <pre><code>// Example code or technical snippet</code></pre>
      </div>
      
      <!-- Comparison Table -->
      <div class="comparison-table">
        <table>
          <thead>
            <tr><th>Feature</th><th>GPT-4</th><th>o3</th></tr>
          </thead>
          <tbody>
            <tr><td>...</td><td>...</td><td>...</td></tr>
          </tbody>
        </table>
      </div>
      
    </div>
  </div>
  
  <!-- Article Footer -->
  <footer class="article-footer">
    <div class="container container--narrow">
      <!-- Tags -->
      <div class="article-tags">
        <a href="/tag/openai/" class="tag">OpenAI</a>
        <a href="/tag/chatgpt/" class="tag">ChatGPT</a>
        <a href="/tag/llms/" class="tag">LLMs</a>
      </div>
      
      <!-- Author Bio -->
      <div class="author-bio">
        <img src="/author-alex.jpg" alt="">
        <div class="author-bio__content">
          <h4>Alex Clearfield</h4>
          <p>AI Correspondent covering the biggest stories in artificial intelligence. 
             Focused on making AI accessible to everyone.</p>
          <div class="author-bio__social">
            <a href="#">Twitter</a>
            <a href="#">LinkedIn</a>
          </div>
        </div>
      </div>
      
      <!-- Newsletter CTA -->
      <div class="article-newsletter">
        <h4>Enjoyed this analysis?</h4>
        <p>Get stories like this delivered weekly.</p>
        <form>
          <input type="email" placeholder="your@email.com">
          <button type="submit">Subscribe</button>
        </form>
      </div>
      
      <!-- Related Posts -->
      <div class="related-posts">
        <h4>Keep Reading</h4>
        <div class="related-grid">
          <!-- 3 related articles -->
        </div>
      </div>
    </div>
  </footer>
</article>
```

---

##  REVIEW PAGE TEMPLATE

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.8

### Complete Review Layout

```html
<!-- Single Review Page -->
<article class="shgr-review" itemscope itemtype="https://schema.org/Review">
  <!-- Review Header -->
  <header class="review-header">
    <div class="container">
      <nav class="breadcrumb">
        <a href="/">Home</a> / 
        <a href="/smart-lighting/">Smart Lighting</a> / 
        <span>Philips Hue Review</span>
      </nav>
      
      <div class="review-header__badges">
        <span class="badge badge--lab-tested">
          <svg><!-- flask --></svg>
          Lab Tested
        </span>
        <span class="badge badge--editors-choice">
           Editor's Choice
        </span>
      </div>
      
      <h1 class="review-title" itemprop="name">
        Philips Hue White & Color Ambiance Starter Kit Review
      </h1>
      
      <p class="review-subtitle">
        Is the premium price worth it? We spent 45 days finding out.
      </p>
      
      <div class="review-meta">
        <div class="review-meta__author">
          <img src="/team/marcus.jpg" alt="Marcus Gear">
          <div class="author-info">
            <span class="author-name">Marcus Gear</span>
            <span class="author-title">Lead Reviewer</span>
          </div>
        </div>
        <div class="review-meta__date">
          <span class="label">Published</span>
          <time datetime="2025-12-10">December 10, 2025</time>
        </div>
        <div class="review-meta__updated">
          <span class="label">Last Updated</span>
          <time datetime="2025-12-15">December 15, 2025</time>
        </div>
      </div>
    </div>
  </header>
  
  <!-- Score Summary Card -->
  <div class="review-score-section">
    <div class="container">
      <div class="score-summary">
        <!-- Main Score -->
        <div class="score-main">
          <div class="score-circle" data-score="9.1">
            <svg class="score-ring">
              <circle class="bg" cx="60" cy="60" r="54"/>
              <circle class="progress" cx="60" cy="60" r="54"/>
            </svg>
            <span class="score-value">9.1</span>
          </div>
          <span class="score-label">Excellent</span>
          <div class="verdict-badge">Editor's Choice</div>
        </div>
        
        <!-- Score Breakdown -->
        <div class="score-breakdown">
          <h4>Score Breakdown</h4>
          <div class="breakdown-items">
            <div class="breakdown-item">
              <span class="item-label">Setup & Installation</span>
              <div class="item-bar">
                <div class="bar-fill" style="--score: 8.5"></div>
              </div>
              <span class="item-score">8.5</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Performance</span>
              <div class="item-bar">
                <div class="bar-fill" style="--score: 9.5"></div>
              </div>
              <span class="item-score">9.5</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Features</span>
              <div class="item-bar">
                <div class="bar-fill" style="--score: 9.5"></div>
              </div>
              <span class="item-score">9.5</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Build Quality</span>
              <div class="item-bar">
                <div class="bar-fill" style="--score: 9.0"></div>
              </div>
              <span class="item-score">9.0</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">App & Software</span>
              <div class="item-bar">
                <div class="bar-fill" style="--score: 9.0"></div>
              </div>
              <span class="item-score">9.0</span>
            </div>
            <div class="breakdown-item">
              <span class="item-label">Value</span>
              <div class="item-bar">
                <div class="bar-fill" style="--score: 8.0"></div>
              </div>
              <span class="item-score">8.0</span>
            </div>
          </div>
        </div>
        
        <!-- Quick Verdict -->
        <div class="quick-verdict">
          <h4>The Verdict</h4>
          <p>
            The Philips Hue system remains the gold standard in smart lighting. 
            Premium pricing is justified by unmatched reliability, ecosystem depth, 
            and continuous improvement.
          </p>
        </div>
        
        <!-- Buy CTA -->
        <div class="buy-cta">
          <span class="price">$179.99</span>
          <a href="#" class="btn btn-primary" rel="sponsored nofollow">
            Check Price at Amazon
          </a>
          <a href="#where-to-buy" class="btn btn-outline">
            More Retailers
          </a>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Pros/Cons -->
  <div class="review-pros-cons">
    <div class="container container--narrow">
      <div class="pros-cons">
        <div class="pros">
          <h4>
            <svg><!-- thumbs up --></svg>
            What We Liked
          </h4>
          <ul>
            <li>Industry-leading color accuracy and brightness</li>
            <li>Rock-solid reliability with local control option</li>
            <li>Massive accessory and bulb ecosystem</li>
            <li>Excellent app with advanced automation</li>
            <li>Matter support for future-proofing</li>
          </ul>
        </div>
        <div class="cons">
          <h4>
            <svg><!-- thumbs down --></svg>
            What Could Be Better
          </h4>
          <ul>
            <li>Premium pricing across the line</li>
            <li>Hub required for full functionality</li>
            <li>Some advanced features locked behind subscription</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Table of Contents -->
  <nav class="review-toc">
    <div class="container container--narrow">
      <h4>In This Review</h4>
      <ol class="toc-list">
        <li><a href="#overview">Overview & Key Specs</a></li>
        <li><a href="#setup">Setup Experience</a></li>
        <li><a href="#performance">Performance Testing</a></li>
        <li><a href="#features">Features Deep Dive</a></li>
        <li><a href="#app">App & Software</a></li>
        <li><a href="#comparison">vs Competition</a></li>
        <li><a href="#verdict">Final Verdict</a></li>
        <li><a href="#faq">FAQ</a></li>
      </ol>
    </div>
  </nav>
  
  <!-- Review Content -->
  <div class="review-content" itemprop="reviewBody">
    <div class="container container--narrow">
      
      <!-- Overview Section -->
      <section id="overview">
        <h2>Overview & Key Specs</h2>
        <p>Introduction and context...</p>
        
        <!-- Key Specs Table -->
        <div class="specs-highlight">
          <table class="specs-table">
            <tbody>
              <tr>
                <th>Bulb Type</th>
                <td>A19 (E26 base)</td>
              </tr>
              <tr>
                <th>Color Range</th>
                <td>16 million colors + 50,000K white</td>
              </tr>
              <tr>
                <th>Brightness</th>
                <td>800 lumens (60W equivalent)</td>
              </tr>
              <tr>
                <th>Connectivity</th>
                <td>Zigbee (requires Bridge)</td>
              </tr>
              <tr>
                <th>Voice Support</th>
                <td>Alexa, Google, Siri, SmartThings</td>
              </tr>
              <tr>
                <th>Warranty</th>
                <td>2 years</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      
      <!-- More content sections... -->
      
      <!-- Comparison Section -->
      <section id="comparison">
        <h2>How It Compares</h2>
        
        <div class="comparison-table-wrapper">
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th class="current">Philips Hue</th>
                <th>LIFX</th>
                <th>Nanoleaf</th>
                <th>Wyze</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Our Score</td>
                <td class="current"><strong>9.1</strong></td>
                <td>8.5</td>
                <td>8.3</td>
                <td>7.8</td>
              </tr>
              <tr>
                <td>Price (Starter)</td>
                <td class="current">$179</td>
                <td>$139</td>
                <td>$99</td>
                <td>$34</td>
              </tr>
              <tr>
                <td>Hub Required</td>
                <td class="current">Yes</td>
                <td>No</td>
                <td>No</td>
                <td>No</td>
              </tr>
              <!-- More rows... -->
            </tbody>
          </table>
        </div>
      </section>
      
      <!-- FAQ Section -->
      <section id="faq">
        <h2>Frequently Asked Questions</h2>
        
        <div class="faq-list" itemscope itemtype="https://schema.org/FAQPage">
          <div class="faq-item" itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">
            <h3 itemprop="name">Do I need the Hue Bridge?</h3>
            <div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">
              <p itemprop="text">
                While Hue bulbs can work via Bluetooth without the Bridge, 
                you'll miss out on advanced features like away-from-home control, 
                automation, and accessories. We recommend the Bridge for most users.
              </p>
            </div>
          </div>
          <!-- More FAQs... -->
        </div>
      </section>
      
    </div>
  </div>
  
  <!-- Where to Buy -->
  <div class="review-where-to-buy" id="where-to-buy">
    <div class="container container--narrow">
      <h3>Where to Buy</h3>
      <div class="retailer-list">
        <a href="#" class="retailer-card" rel="sponsored nofollow">
          <img src="/retailers/amazon.svg" alt="Amazon">
          <span class="retailer-name">Amazon</span>
          <span class="retailer-price">$179.99</span>
          <span class="btn btn-sm">Check Price</span>
        </a>
        <a href="#" class="retailer-card" rel="sponsored nofollow">
          <img src="/retailers/bestbuy.svg" alt="Best Buy">
          <span class="retailer-name">Best Buy</span>
          <span class="retailer-price">$179.99</span>
          <span class="btn btn-sm">Check Price</span>
        </a>
        <a href="#" class="retailer-card" rel="sponsored nofollow">
          <img src="/retailers/homedepot.svg" alt="Home Depot">
          <span class="retailer-name">Home Depot</span>
          <span class="retailer-price">$179.99</span>
          <span class="btn btn-sm">Check Price</span>
        </a>
      </div>
      <p class="affiliate-note">
        * We earn commission on qualifying purchases. Prices updated daily.
      </p>
    </div>
  </div>
  
  <!-- Related Reviews -->
  <div class="review-related">
    <div class="container">
      <h3>Related Reviews</h3>
      <div class="related-grid">
        <!-- Related product cards -->
      </div>
    </div>
  </div>
</article>
```

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

### Single Review Layout

```html
<!-- Product Review Page -->
<article class="wgr-review" itemscope itemtype="https://schema.org/Review">
  <!-- Review Header -->
  <header class="review-header">
    <div class="container">
      <div class="review-header__breadcrumb">
        <a href="/">Home</a> / 
        <a href="/smartwatches/">Smartwatches</a> / 
        <a href="/smartwatches/apple-watch/">Apple Watch</a>
      </div>
      
      <div class="review-header__meta">
        <span class="review-category">Smartwatch Review</span>
        <span class="review-date">Updated: Dec 15, 2025</span>
      </div>
      
      <h1 class="review-title" itemprop="name">
        Apple Watch Series 10 Review: The Thinnest Yet
      </h1>
      
      <p class="review-subtitle">
        Significant display upgrades and the slimmest design ever, 
        but is it worth upgrading from Series 9?
      </p>
    </div>
  </header>
  
  <!-- Review Score Card -->
  <div class="review-score-card">
    <div class="container">
      <div class="score-card">
        <div class="score-card__main">
          <div class="score-circle" data-score="9.2">
            <svg class="score-ring"><!-- animated ring --></svg>
            <span class="score-value">9.2</span>
          </div>
          <span class="score-label">Excellent</span>
          <span class="score-badge">Editor's Choice</span>
        </div>
        
        <div class="score-card__breakdown">
          <div class="score-item">
            <span class="score-item__label">Design</span>
            <div class="score-bar" data-score="9.5"></div>
            <span class="score-item__value">9.5</span>
          </div>
          <div class="score-item">
            <span class="score-item__label">Display</span>
            <div class="score-bar" data-score="9.8"></div>
            <span class="score-item__value">9.8</span>
          </div>
          <div class="score-item">
            <span class="score-item__label">Features</span>
            <div class="score-bar" data-score="9.0"></div>
            <span class="score-item__value">9.0</span>
          </div>
          <div class="score-item">
            <span class="score-item__label">Battery</span>
            <div class="score-bar" data-score="7.5"></div>
            <span class="score-item__value">7.5</span>
          </div>
          <div class="score-item">
            <span class="score-item__label">Value</span>
            <div class="score-bar" data-score="8.5"></div>
            <span class="score-item__value">8.5</span>
          </div>
        </div>
        
        <div class="score-card__verdict">
          <h4>The Verdict</h4>
          <p>
            The Apple Watch Series 10 is the best smartwatch for iPhone users, 
            with meaningful upgrades in display size and thinness.
          </p>
        </div>
        
        <div class="score-card__cta">
          <a href="#" class="btn btn-primary" rel="sponsored">
            Check Price at Amazon
          </a>
          <span class="price-note">Starting at $399</span>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Quick Specs -->
  <div class="review-quick-specs">
    <div class="container">
      <div class="quick-specs">
        <div class="spec-item">
          <span class="spec-label">Display</span>
          <span class="spec-value">1.96" OLED</span>
        </div>
        <div class="spec-item">
          <span class="spec-label">Battery</span>
          <span class="spec-value">18 hours</span>
        </div>
        <div class="spec-item">
          <span class="spec-label">Water Resist</span>
          <span class="spec-value">50m</span>
        </div>
        <div class="spec-item">
          <span class="spec-label">GPS</span>
          <span class="spec-value">Yes</span>
        </div>
        <div class="spec-item">
          <span class="spec-label">Weight</span>
          <span class="spec-value">36.4g</span>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Pros/Cons -->
  <div class="review-pros-cons">
    <div class="container">
      <div class="pros-cons-grid">
        <div class="pros">
          <h4> What We Loved</h4>
          <ul>
            <li>Gorgeous larger display</li>
            <li>Incredibly thin design</li>
            <li>Excellent fitness tracking</li>
            <li>Bright always-on mode</li>
          </ul>
        </div>
        <div class="cons">
          <h4> What Could Be Better</h4>
          <ul>
            <li>Still only 18-hour battery</li>
            <li>Premium price</li>
            <li>Minor upgrade from Series 9</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Review Content -->
  <div class="review-content" itemprop="reviewBody">
    <div class="container container--narrow">
      <!-- Table of Contents -->
      <nav class="review-toc">
        <h4>In This Review</h4>
        <ol>
          <li><a href="#design">Design & Build</a></li>
          <li><a href="#display">Display</a></li>
          <li><a href="#fitness">Fitness Tracking</a></li>
          <li><a href="#health">Health Features</a></li>
          <li><a href="#battery">Battery Life</a></li>
          <li><a href="#comparison">vs Competition</a></li>
          <li><a href="#verdict">Final Verdict</a></li>
        </ol>
      </nav>
      
      <!-- Content Sections -->
      <section id="design">
        <h2>Design & Build Quality</h2>
        <p>Content...</p>
        
        <!-- Image Gallery -->
        <div class="review-gallery">
          <img src="..." alt="">
        </div>
      </section>
      
      <!-- ... more sections ... -->
      
      <!-- Spec Table -->
      <section id="specs">
        <h2>Full Specifications</h2>
        <table class="specs-table">
          <tbody>
            <tr>
              <th>Display</th>
              <td>1.96" LTPO OLED, 2000 nits</td>
            </tr>
            <tr>
              <th>Processor</th>
              <td>S10 SiP</td>
            </tr>
            <!-- ... -->
          </tbody>
        </table>
      </section>
      
    </div>
  </div>
  
  <!-- Related Content -->
  <div class="review-related">
    <div class="container">
      <h3>Related Reviews</h3>
      <div class="related-grid">
        <!-- Related review cards -->
      </div>
    </div>
  </div>
</article>
```

---

##  REQUIRED PLUGINS

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.6

### Essential Stack

```
THEME:
 Blocksy Theme (free)
 Blocksy Companion Pro (premium features)

PAGE BUILDER:
 Elementor Pro (already installed)

SEO:
 RankMath Pro (already installed)
  - Configure schema for Organization
  - Set up Local SEO if applicable
  - Configure News Sitemap

PERFORMANCE:
 LiteSpeed Cache (already installed)
  - Enable page cache
  - Enable browser cache
  - Enable LazyLoad
  - Minify CSS/JS

SECURITY:
 Wordfence (add whitelist rules for tools)
 Complianz GDPR (already installed)

CONVERSION:
○ WPForms Pro (forms + surveys)
○ OptinMonster or Convert Pro (popups/lead capture)

FUNCTIONALITY:
○ MegaMenu Pro (for ecosystem mega menu)
○ TablePress (comparison tables)
○ WP Show Posts (grid layouts)
○ Schema Pro (additional schema types)

ANALYTICS:
 Site Kit by Google (already installed)
○ MonsterInsights Pro (enhanced analytics)
```

---

## ️ HOMEPAGE DESIGN

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.4

### Hero Section - News Ticker + Featured

```html
<!-- Hero: Breaking News + Featured Story -->
<section class="clear-hero">
  <!-- Live Ticker Bar -->
  <div class="clear-ticker">
    <span class="ticker-label">
      <span class="pulse-dot"></span>
      LIVE
    </span>
    <div class="ticker-content">
      <marquee-component>
        <!-- Auto-populated from Breaking News category -->
      </marquee-component>
    </div>
  </div>
  
  <!-- Featured Story -->
  <div class="clear-hero__featured">
    <div class="clear-hero__background">
      <div class="neural-grid"></div>
      <div class="glow-effect"></div>
    </div>
    
    <div class="container">
      <article class="featured-story">
        <span class="featured-story__category cat-analysis">
          Analysis
        </span>
        <h1 class="featured-story__title">
          <a href="#">
            OpenAI's o3 Changes Everything--Here's What It Means for You
          </a>
        </h1>
        <p class="featured-story__excerpt">
          The new reasoning model isn't just another GPT. We break down 
          why this release matters and what it signals for AI's future.
        </p>
        <div class="featured-story__meta">
          <img src="/author-alex.jpg" alt="Alex Clearfield" class="author-avatar">
          <span class="author-name">Alex Clearfield</span>
          <span class="separator">·</span>
          <time datetime="2025-12-16">Dec 16, 2025</time>
          <span class="separator">·</span>
          <span class="read-time">8 min read</span>
        </div>
      </article>
    </div>
  </div>
</section>
```

### News Grid Layout

```html
<!-- Latest News Grid -->
<section class="clear-news-grid">
  <div class="container">
    <header class="section-header">
      <h2>Latest Stories</h2>
      <div class="filter-tabs">
        <button class="filter-tab active" data-filter="all">All</button>
        <button class="filter-tab" data-filter="analysis">Analysis</button>
        <button class="filter-tab" data-filter="industry">Industry</button>
        <button class="filter-tab" data-filter="research">Research</button>
        <button class="filter-tab" data-filter="ethics">Ethics</button>
      </div>
    </header>
    
    <div class="news-grid">
      <!-- Large Feature Card -->
      <article class="news-card news-card--large">
        <div class="news-card__image">
          <img src="..." alt="">
          <span class="news-card__category cat-breaking">Breaking</span>
        </div>
        <div class="news-card__content">
          <h3 class="news-card__title">
            <a href="#">Article Title Here</a>
          </h3>
          <p class="news-card__excerpt">Brief excerpt...</p>
          <div class="news-card__meta">
            <time>2 hours ago</time>
            <span>5 min read</span>
          </div>
        </div>
      </article>
      
      <!-- Standard Cards (x5) -->
      <article class="news-card">
        <!-- Same structure, smaller -->
      </article>
    </div>
    
    <div class="news-grid__footer">
      <a href="/latest/" class="btn btn-outline">
        View All Stories
        <svg><!-- arrow --></svg>
      </a>
    </div>
  </div>
</section>
```

### Explained Section - Educational Hub

```html
<!-- AI Explained Section -->
<section class="clear-explained">
  <div class="container">
    <header class="section-header section-header--centered">
      <span class="section-badge">New to AI?</span>
      <h2>AI Explained</h2>
      <p>Complex concepts, human language. Start here.</p>
    </header>
    
    <div class="explained-grid">
      <!-- AI 101 Card -->
      <article class="explained-card explained-card--featured">
        <div class="explained-card__icon">
          <svg><!-- graduation cap --></svg>
        </div>
        <h3>AI 101: The Complete Beginner's Guide</h3>
        <p>Everything you need to understand AI, starting from zero.</p>
        <ul class="explained-card__topics">
          <li>What is AI, really?</li>
          <li>How do chatbots work?</li>
          <li>AI vs ML vs Deep Learning</li>
          <li>The companies behind AI</li>
        </ul>
        <a href="/ai-101/" class="btn btn-primary">Start Learning</a>
      </article>
      
      <!-- Glossary Card -->
      <article class="explained-card">
        <div class="explained-card__icon">
          <svg><!-- book --></svg>
        </div>
        <h3>AI Glossary</h3>
        <p>LLM? RAG? AGI? We define every term you'll encounter.</p>
        <a href="/glossary/" class="explained-card__link">Browse Terms →</a>
      </article>
      
      <!-- How AI Works Card -->
      <article class="explained-card">
        <div class="explained-card__icon">
          <svg><!-- brain --></svg>
        </div>
        <h3>How AI Actually Works</h3>
        <p>The mechanics behind the magic, explained simply.</p>
        <a href="/how-ai-works/" class="explained-card__link">Learn More →</a>
      </article>
      
      <!-- Timeline Card -->
      <article class="explained-card">
        <div class="explained-card__icon">
          <svg><!-- timeline --></svg>
        </div>
        <h3>AI Timeline</h3>
        <p>From Turing to transformers--the history of AI.</p>
        <a href="/ai-timeline/" class="explained-card__link">Explore →</a>
      </article>
    </div>
  </div>
</section>
```

### Newsletter Section

```html
<!-- Newsletter CTA -->
<section class="clear-newsletter">
  <div class="container">
    <div class="newsletter-card">
      <div class="newsletter-card__content">
        <span class="newsletter-card__badge">Free Weekly Newsletter</span>
        <h2>AI News, Decoded</h2>
        <p>
          Every Saturday: The week's biggest AI stories explained 
          in plain English. No jargon, no hype, just clarity.
        </p>
        
        <form class="newsletter-form" action="#" method="post">
          <div class="newsletter-form__field">
            <input type="email" placeholder="your@email.com" required>
            <button type="submit" class="btn btn-primary">
              Subscribe
            </button>
          </div>
          <p class="newsletter-form__note">
            Join 10,000+ readers. Unsubscribe anytime.
          </p>
        </form>
      </div>
      
      <div class="newsletter-card__visual">
        <div class="newsletter-preview">
          <!-- Animated newsletter preview mockup -->
        </div>
      </div>
    </div>
  </div>
</section>
```

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

### Hero Section

```html
<!-- Hero: Authority + Search Focus -->
<section class="shgr-hero">
  <div class="shgr-hero__background">
    <div class="hero-grid"></div>
    <div class="hero-glow"></div>
  </div>
  
  <div class="shgr-hero__content">
    <div class="container">
      <span class="shgr-hero__badge">
        <svg><!-- lab flask icon --></svg>
        500+ Products Tested
      </span>
      
      <h1 class="shgr-hero__title">
        Smart Home Reviews<br>
        <span class="gradient-text">You Can Trust</span>
      </h1>
      
      <p class="shgr-hero__subtitle">
        Lab-tested reviews with 30+ day testing periods. 
        Real performance data. Honest verdicts.
      </p>
      
      <!-- Search Bar -->
      <div class="shgr-hero__search">
        <form class="hero-search" action="/search/" method="get">
          <input type="text" name="s" placeholder="Search products, brands, or categories...">
          <button type="submit">
            <svg><!-- search icon --></svg>
            Search
          </button>
        </form>
        <div class="search-shortcuts">
          <span>Popular:</span>
          <a href="/best-smart-lights/">Smart Lights</a>
          <a href="/best-video-doorbells/">Video Doorbells</a>
          <a href="/best-smart-thermostats/">Thermostats</a>
        </div>
      </div>
      
      <!-- Trust Indicators -->
      <div class="shgr-hero__trust">
        <div class="trust-item">
          <span class="trust-number">500+</span>
          <span class="trust-label">Products Reviewed</span>
        </div>
        <div class="trust-item">
          <span class="trust-number">30+</span>
          <span class="trust-label">Day Test Periods</span>
        </div>
        <div class="trust-item">
          <span class="trust-number">10K+</span>
          <span class="trust-label">Monthly Readers</span>
        </div>
      </div>
    </div>
  </div>
</section>
```

### Category Browser

```html
<!-- Browse by Category -->
<section class="shgr-categories">
  <div class="container">
    <header class="section-header">
      <h2 class="section-title">Browse by Category</h2>
      <p>Find the perfect device for every room</p>
    </header>
    
    <div class="category-grid category-grid--6col">
      <!-- Smart Lighting -->
      <a href="/smart-lighting/" class="category-tile">
        <div class="category-tile__icon">
          <svg><!-- lightbulb icon --></svg>
        </div>
        <h3>Smart Lighting</h3>
        <span class="category-tile__count">87 Reviews</span>
      </a>
      
      <!-- Smart Security -->
      <a href="/smart-security/" class="category-tile">
        <div class="category-tile__icon">
          <svg><!-- shield icon --></svg>
        </div>
        <h3>Smart Security</h3>
        <span class="category-tile__count">63 Reviews</span>
      </a>
      
      <!-- Climate Control -->
      <a href="/climate-control/" class="category-tile">
        <div class="category-tile__icon">
          <svg><!-- thermometer icon --></svg>
        </div>
        <h3>Climate Control</h3>
        <span class="category-tile__count">42 Reviews</span>
      </a>
      
      <!-- Smart Speakers -->
      <a href="/smart-speakers/" class="category-tile">
        <div class="category-tile__icon">
          <svg><!-- speaker icon --></svg>
        </div>
        <h3>Smart Speakers</h3>
        <span class="category-tile__count">38 Reviews</span>
      </a>
      
      <!-- Smart Displays -->
      <a href="/smart-displays/" class="category-tile">
        <div class="category-tile__icon">
          <svg><!-- display icon --></svg>
        </div>
        <h3>Smart Displays</h3>
        <span class="category-tile__count">24 Reviews</span>
      </a>
      
      <!-- Smart Hubs -->
      <a href="/smart-hubs/" class="category-tile">
        <div class="category-tile__icon">
          <svg><!-- hub icon --></svg>
        </div>
        <h3>Smart Hubs</h3>
        <span class="category-tile__count">18 Reviews</span>
      </a>
    </div>
  </div>
</section>
```

### Editor's Choice Showcase

```html
<!-- Editor's Choice Products -->
<section class="shgr-editors-choice">
  <div class="container">
    <header class="section-header">
      <span class="section-badge"> Top Picks</span>
      <h2 class="section-title">Editor's Choice</h2>
      <p>Our highest-rated products across all categories</p>
    </header>
    
    <div class="editors-grid">
      <!-- Featured Product Card -->
      <article class="product-card product-card--featured">
        <div class="product-card__badges">
          <span class="badge badge--editors-choice">Editor's Choice</span>
        </div>
        <div class="product-card__image">
          <img src="..." alt="Philips Hue Starter Kit">
        </div>
        <div class="product-card__content">
          <span class="product-card__category">Smart Lighting</span>
          <h3 class="product-card__title">
            <a href="#">Philips Hue White & Color Starter Kit</a>
          </h3>
          <div class="product-card__rating">
            <div class="rating-badge rating-badge--excellent">9.1</div>
            <span class="rating-label">Excellent</span>
          </div>
          <p class="product-card__excerpt">
            The gold standard in smart lighting. Unmatched ecosystem and reliability.
          </p>
          <div class="product-card__footer">
            <span class="product-card__price">$179.99</span>
            <a href="#" class="btn btn-primary btn-sm" rel="sponsored">
              View Deal
            </a>
          </div>
        </div>
      </article>
      
      <!-- More product cards... -->
    </div>
    
    <div class="section-footer">
      <a href="/editors-choice/" class="btn btn-outline">
        View All Editor's Choice
      </a>
    </div>
  </div>
</section>
```

### Latest Reviews Grid

```html
<!-- Latest Reviews -->
<section class="shgr-latest">
  <div class="container">
    <header class="section-header">
      <h2 class="section-title">Latest Reviews</h2>
      <div class="section-tabs">
        <button class="tab active" data-filter="all">All</button>
        <button class="tab" data-filter="lighting">Lighting</button>
        <button class="tab" data-filter="security">Security</button>
        <button class="tab" data-filter="climate">Climate</button>
      </div>
    </header>
    
    <div class="reviews-grid">
      <!-- Review Card -->
      <article class="review-card">
        <a href="#" class="review-card__link">
          <div class="review-card__image">
            <img src="..." alt="">
            <div class="review-card__score">
              <span>8.7</span>
            </div>
          </div>
          <div class="review-card__content">
            <span class="review-card__category">Video Doorbell</span>
            <h3 class="review-card__title">Ring Video Doorbell Pro 2 Review</h3>
            <div class="review-card__meta">
              <span class="date">Dec 14, 2025</span>
              <span class="read-time">10 min read</span>
            </div>
          </div>
        </a>
      </article>
      
      <!-- More cards... -->
    </div>
    
    <div class="section-footer">
      <a href="/reviews/" class="btn btn-outline">
        All Reviews
        <svg><!-- arrow --></svg>
      </a>
    </div>
  </div>
</section>
```

### Lab Section (Trust Builder)

```html
<!-- How We Test Section -->
<section class="shgr-lab">
  <div class="container">
    <div class="lab-showcase">
      <div class="lab-showcase__content">
        <span class="section-badge"> Our Process</span>
        <h2 class="section-title">The SHGR Testing Lab</h2>
        <p class="lab-showcase__intro">
          Every product goes through our rigorous multi-week testing protocol. 
          No quick impressions--real-world performance data.
        </p>
        
        <div class="lab-process">
          <div class="process-step">
            <span class="step-number">01</span>
            <h4>Unboxing & Setup</h4>
            <p>We document the entire setup process, noting any friction points.</p>
          </div>
          <div class="process-step">
            <span class="step-number">02</span>
            <h4>30-Day Testing</h4>
            <p>Products live in our test home for a minimum of 30 days.</p>
          </div>
          <div class="process-step">
            <span class="step-number">03</span>
            <h4>Benchmark Testing</h4>
            <p>Standardized tests for response time, reliability, and accuracy.</p>
          </div>
          <div class="process-step">
            <span class="step-number">04</span>
            <h4>Final Scoring</h4>
            <p>Weighted scoring across 6 categories for fair comparison.</p>
          </div>
        </div>
        
        <a href="/how-we-test/" class="btn btn-secondary">
          Learn About Our Testing
        </a>
      </div>
      
      <div class="lab-showcase__visual">
        <div class="lab-stats">
          <div class="stat">
            <span class="stat-value">30+</span>
            <span class="stat-label">Day Test Period</span>
          </div>
          <div class="stat">
            <span class="stat-value">6</span>
            <span class="stat-label">Scoring Categories</span>
          </div>
          <div class="stat">
            <span class="stat-value">100%</span>
            <span class="stat-label">Real-World Testing</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>
```

### Buying Guides Section

```html
<!-- Buying Guides -->
<section class="shgr-guides">
  <div class="container">
    <header class="section-header">
      <h2 class="section-title">Buying Guides</h2>
      <p>Our expert picks, updated monthly</p>
    </header>
    
    <div class="guides-grid">
      <!-- Guide Card -->
      <article class="guide-card">
        <div class="guide-card__header">
          <span class="guide-card__date">Updated Dec 2025</span>
        </div>
        <div class="guide-card__content">
          <h3>Best Smart Lights 2025</h3>
          <p>From budget bulbs to premium systems</p>
          <div class="guide-card__picks">
            <div class="pick">
              <span class="pick-label">Top Pick</span>
              <span class="pick-product">Philips Hue</span>
            </div>
            <div class="pick">
              <span class="pick-label">Budget</span>
              <span class="pick-product">Wyze Bulb</span>
            </div>
          </div>
          <a href="/best-smart-lights/" class="guide-card__link">
            Read Guide →
          </a>
        </div>
      </article>
      
      <!-- More guides... -->
    </div>
  </div>
</section>
```

---

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.4

### Hero Section

```html
<!-- Hero: Full-viewport immersive experience -->
<section class="haven-hero" data-aos="fade-up">
  <div class="haven-hero__background">
    <video autoplay muted loop playsinline>
      <source src="/videos/smart-home-ambient.mp4" type="video/mp4">
    </video>
    <div class="haven-hero__overlay"></div>
  </div>
  
  <div class="haven-hero__content">
    <span class="haven-hero__badge">
      <span class="pulse-dot"></span>
      Trusted by 50,000+ Smart Home Enthusiasts
    </span>
    
    <h1 class="haven-hero__title">
      Your Home,<br>
      <span class="gradient-text">Intelligently Connected</span>
    </h1>
    
    <p class="haven-hero__subtitle">
      Expert guides and honest reviews to help you build the smart home 
      of your dreams--no engineering degree required.
    </p>
    
    <div class="haven-hero__cta">
      <a href="/getting-started/" class="btn btn-primary btn-lg">
        Start Your Journey
        <svg><!-- arrow icon --></svg>
      </a>
      <a href="/ecosystem-quiz/" class="btn btn-outline btn-lg">
        Take the Quiz
      </a>
    </div>
    
    <div class="haven-hero__ecosystems">
      <span>Works with:</span>
      <div class="ecosystem-logos">
        <img src="/icons/alexa.svg" alt="Amazon Alexa">
        <img src="/icons/homekit.svg" alt="Apple HomeKit">
        <img src="/icons/google-home.svg" alt="Google Home">
        <img src="/icons/smartthings.svg" alt="SmartThings">
        <img src="/icons/matter.svg" alt="Matter">
      </div>
    </div>
  </div>
  
  <div class="haven-hero__scroll-indicator">
    <span>Scroll to explore</span>
    <svg><!-- animated down arrow --></svg>
  </div>
</section>
```

### Ecosystem Selection Cards

```html
<!-- Ecosystem Cards: Interactive selection -->
<section class="haven-ecosystems">
  <div class="container">
    <header class="section-header">
      <h2>Choose Your Ecosystem</h2>
      <p>Every smart home starts with choosing the right platform. Find yours.</p>
    </header>
    
    <div class="ecosystem-grid">
      <!-- Alexa Card -->
      <article class="ecosystem-card ecosystem-card--alexa" data-ecosystem="alexa">
        <div class="ecosystem-card__icon">
          <img src="/icons/alexa-glow.svg" alt="">
        </div>
        <h3 class="ecosystem-card__title">Amazon Alexa</h3>
        <p class="ecosystem-card__desc">
          Largest device selection. Best for voice control enthusiasts.
        </p>
        <ul class="ecosystem-card__features">
          <li>10,000+ compatible devices</li>
          <li>Advanced routines</li>
          <li>Budget-friendly options</li>
        </ul>
        <a href="/alexa-ecosystem/" class="ecosystem-card__link">
          Explore Alexa
          <svg><!-- arrow --></svg>
        </a>
      </article>
      
      <!-- HomeKit Card -->
      <article class="ecosystem-card ecosystem-card--homekit" data-ecosystem="homekit">
        <div class="ecosystem-card__icon">
          <img src="/icons/homekit-glow.svg" alt="">
        </div>
        <h3 class="ecosystem-card__title">Apple HomeKit</h3>
        <p class="ecosystem-card__desc">
          Premium privacy. Best for Apple households.
        </p>
        <ul class="ecosystem-card__features">
          <li>End-to-end encryption</li>
          <li>Siri integration</li>
          <li>Seamless Apple experience</li>
        </ul>
        <a href="/apple-homekit-ecosystem/" class="ecosystem-card__link">
          Explore HomeKit
          <svg><!-- arrow --></svg>
        </a>
      </article>
      
      <!-- Google Card -->
      <article class="ecosystem-card ecosystem-card--google" data-ecosystem="google">
        <div class="ecosystem-card__icon">
          <img src="/icons/google-glow.svg" alt="">
        </div>
        <h3 class="ecosystem-card__title">Google Home</h3>
        <p class="ecosystem-card__desc">
          Smartest assistant. Best for information seekers.
        </p>
        <ul class="ecosystem-card__features">
          <li>AI-powered assistant</li>
          <li>Nest ecosystem</li>
          <li>Multi-user support</li>
        </ul>
        <a href="/google-home-ecosystem/" class="ecosystem-card__link">
          Explore Google
          <svg><!-- arrow --></svg>
        </a>
      </article>
      
      <!-- SmartThings Card -->
      <article class="ecosystem-card ecosystem-card--smartthings" data-ecosystem="smartthings">
        <div class="ecosystem-card__icon">
          <img src="/icons/smartthings-glow.svg" alt="">
        </div>
        <h3 class="ecosystem-card__title">SmartThings</h3>
        <p class="ecosystem-card__desc">
          Most flexible. Best for power users.
        </p>
        <ul class="ecosystem-card__features">
          <li>Cross-platform hub</li>
          <li>Advanced automation</li>
          <li>Edge computing</li>
        </ul>
        <a href="/smartthings-ecosystem/" class="ecosystem-card__link">
          Explore SmartThings
          <svg><!-- arrow --></svg>
        </a>
      </article>
    </div>
    
    <div class="ecosystem-cta">
      <p>Not sure which is right for you?</p>
      <a href="/ecosystem-quiz/" class="btn btn-secondary">
        Take Our 2-Minute Quiz
      </a>
    </div>
  </div>
</section>
```

### Free Resources Section

```html
<!-- Lead Magnets: Strategic resource delivery -->
<section class="haven-resources">
  <div class="container">
    <header class="section-header">
      <span class="section-badge">Free Downloads</span>
      <h2>Tools to Kickstart Your Smart Home</h2>
      <p>Grab these essential resources--on us.</p>
    </header>
    
    <div class="resource-grid">
      <article class="resource-card">
        <div class="resource-card__preview">
          <img src="/images/starter-guide-preview.png" alt="Smart Home Starter Guide">
        </div>
        <div class="resource-card__content">
          <span class="resource-card__type">PDF Guide</span>
          <h3>Smart Home Starter Guide</h3>
          <p>The complete beginner's blueprint to planning, budgeting, and building your first smart home.</p>
          <a href="/smart-home-starter-guide-3/" class="btn btn-primary">
            Download Free
          </a>
        </div>
      </article>
      
      <article class="resource-card">
        <div class="resource-card__preview">
          <img src="/images/alexa-cheatsheet-preview.png" alt="Alexa Commands Cheat Sheet">
        </div>
        <div class="resource-card__content">
          <span class="resource-card__type">Cheat Sheet</span>
          <h3>Alexa Commands Cheat Sheet</h3>
          <p>200+ voice commands organized by category. Print it, stick it on your fridge.</p>
          <a href="/alexa-commands-cheat-sheet-2/" class="btn btn-primary">
            Download Free
          </a>
        </div>
      </article>
      
      <article class="resource-card">
        <div class="resource-card__preview">
          <img src="/images/budget-planner-preview.png" alt="Smart Home Budget Planner">
        </div>
        <div class="resource-card__content">
          <span class="resource-card__type">Spreadsheet</span>
          <h3>Budget Planner</h3>
          <p>Plan your smart home investment room by room. Avoid overspending.</p>
          <a href="/smart-home-budget-planner-2/" class="btn btn-primary">
            Download Free
          </a>
        </div>
      </article>
      
      <article class="resource-card">
        <div class="resource-card__preview">
          <img src="/images/security-checklist-preview.png" alt="Security Checklist">
        </div>
        <div class="resource-card__content">
          <span class="resource-card__type">Checklist</span>
          <h3>Security Checklist</h3>
          <p>Protect your smart home from hackers with this comprehensive security audit.</p>
          <a href="/smart-home-security-checklist-2/" class="btn btn-primary">
            Download Free
          </a>
        </div>
      </article>
    </div>
  </div>
</section>
```

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

### Hero Section (Enhanced)

```html
<!-- WGR Hero - Existing structure enhanced -->
<section class="wgr-hero">
  <div class="wgr-hero__background">
    <video autoplay muted loop playsinline>
      <source src="/videos/wearables-showcase.mp4" type="video/mp4">
    </video>
    <div class="wgr-hero__overlay"></div>
    <div class="wgr-hero__pulse"></div>
  </div>
  
  <div class="wgr-hero__content">
    <span class="wgr-hero__badge">
      <span class="pulse-dot"></span>
      Trusted by 25,000+ Fitness Enthusiasts
    </span>
    
    <h1 class="wgr-hero__title">
      Find Your Perfect<br>
      <span class="gradient-text">Wearable</span>
    </h1>
    
    <p class="wgr-hero__subtitle">
      Data-driven reviews that cut through the hype. 
      Real tests. Real workouts. Real results.
    </p>
    
    <div class="wgr-hero__search">
      <div class="search-box">
        <input type="text" placeholder="Search devices, brands, or features...">
        <button type="submit">
          <svg><!-- search icon --></svg>
        </button>
      </div>
      <div class="search-suggestions">
        <span>Popular:</span>
        <a href="/best-smartwatches-2025/">Best Smartwatches</a>
        <a href="/apple-watch-vs-galaxy-watch/">Apple vs Samsung</a>
        <a href="/best-fitness-trackers/">Fitness Trackers</a>
      </div>
    </div>
  </div>
</section>
```

### Category Cards Grid

```html
<!-- Product Categories -->
<section class="wgr-categories">
  <div class="container">
    <header class="section-header">
      <h2>Browse by Category</h2>
      <p>Find the right wearable for your lifestyle</p>
    </header>
    
    <div class="category-grid">
      <!-- Smartwatches -->
      <a href="/smartwatches/" class="category-card category-card--smartwatch">
        <div class="category-card__icon">
          <svg><!-- watch icon --></svg>
        </div>
        <h3>Smartwatches</h3>
        <span class="category-card__count">47 Reviews</span>
        <div class="category-card__brands">
          <img src="/brands/apple.svg" alt="Apple">
          <img src="/brands/samsung.svg" alt="Samsung">
          <img src="/brands/garmin.svg" alt="Garmin">
        </div>
      </a>
      
      <!-- Fitness Trackers -->
      <a href="/fitness-trackers/" class="category-card category-card--fitness">
        <div class="category-card__icon">
          <svg><!-- activity icon --></svg>
        </div>
        <h3>Fitness Trackers</h3>
        <span class="category-card__count">32 Reviews</span>
        <div class="category-card__brands">
          <img src="/brands/fitbit.svg" alt="Fitbit">
          <img src="/brands/whoop.svg" alt="Whoop">
          <img src="/brands/oura.svg" alt="Oura">
        </div>
      </a>
      
      <!-- Health Monitors -->
      <a href="/health-monitors/" class="category-card category-card--health">
        <div class="category-card__icon">
          <svg><!-- heart icon --></svg>
        </div>
        <h3>Health Monitors</h3>
        <span class="category-card__count">18 Reviews</span>
        <div class="category-card__brands">
          <img src="/brands/withings.svg" alt="Withings">
          <img src="/brands/dexcom.svg" alt="Dexcom">
        </div>
      </a>
      
      <!-- Audio -->
      <a href="/audio-wearables/" class="category-card category-card--audio">
        <div class="category-card__icon">
          <svg><!-- headphone icon --></svg>
        </div>
        <h3>Audio Wearables</h3>
        <span class="category-card__count">28 Reviews</span>
        <div class="category-card__brands">
          <img src="/brands/apple.svg" alt="AirPods">
          <img src="/brands/sony.svg" alt="Sony">
          <img src="/brands/bose.svg" alt="Bose">
        </div>
      </a>
      
      <!-- Sports -->
      <a href="/sports-wearables/" class="category-card category-card--sports">
        <div class="category-card__icon">
          <svg><!-- running icon --></svg>
        </div>
        <h3>Sports Wearables</h3>
        <span class="category-card__count">24 Reviews</span>
        <div class="category-card__brands">
          <img src="/brands/garmin.svg" alt="Garmin">
          <img src="/brands/coros.svg" alt="Coros">
          <img src="/brands/polar.svg" alt="Polar">
        </div>
      </a>
    </div>
  </div>
</section>
```

### Featured Reviews Carousel

```html
<!-- Latest & Featured Reviews -->
<section class="wgr-featured">
  <div class="container">
    <header class="section-header">
      <h2>Latest Reviews</h2>
      <a href="/reviews/" class="section-link">All Reviews →</a>
    </header>
    
    <div class="review-carousel">
      <!-- Review Card -->
      <article class="review-card">
        <div class="review-card__image">
          <img src="..." alt="Apple Watch Series 10">
          <span class="review-card__badge">Editor's Choice</span>
          <div class="review-card__rating">
            <span class="rating-score">9.2</span>
          </div>
        </div>
        <div class="review-card__content">
          <span class="review-card__category">Smartwatches</span>
          <h3 class="review-card__title">
            <a href="#">Apple Watch Series 10 Review</a>
          </h3>
          <p class="review-card__excerpt">
            The thinnest Apple Watch yet brings significant display upgrades...
          </p>
          <div class="review-card__meta">
            <span class="date">Dec 15, 2025</span>
            <span class="read-time">12 min read</span>
          </div>
        </div>
      </article>
      
      <!-- More cards... -->
    </div>
  </div>
</section>
```

### Comparison Tool Preview

```html
<!-- Comparison Tool CTA -->
<section class="wgr-compare-cta">
  <div class="container">
    <div class="compare-preview">
      <div class="compare-preview__content">
        <span class="section-badge">Interactive Tool</span>
        <h2>Compare Wearables Side-by-Side</h2>
        <p>
          Can't decide between two devices? Our comparison tool lets you 
          compare specs, features, and our ratings in one view.
        </p>
        <a href="/compare/" class="btn btn-primary btn-lg">
          Start Comparing
          <svg><!-- arrow --></svg>
        </a>
      </div>
      
      <div class="compare-preview__visual">
        <!-- Animated comparison mockup -->
        <div class="compare-mockup">
          <div class="compare-device compare-device--left">
            <img src="/products/apple-watch-10.png" alt="">
            <span>Apple Watch 10</span>
          </div>
          <div class="compare-vs">VS</div>
          <div class="compare-device compare-device--right">
            <img src="/products/galaxy-watch-7.png" alt="">
            <span>Galaxy Watch 7</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>
```

### Buying Guides Section

```html
<!-- Buying Guides -->
<section class="wgr-guides">
  <div class="container">
    <header class="section-header">
      <h2>Buying Guides</h2>
      <p>Our top picks, updated monthly</p>
    </header>
    
    <div class="guides-grid">
      <!-- Best Smartwatches -->
      <article class="guide-card">
        <div class="guide-card__image">
          <img src="..." alt="">
          <span class="guide-card__tag">Updated Dec 2025</span>
        </div>
        <div class="guide-card__content">
          <h3>Best Smartwatches 2025</h3>
          <p>Our top picks after testing 30+ devices</p>
          <div class="guide-card__preview">
            <span class="preview-item">
              <strong>#1</strong> Apple Watch Series 10
            </span>
            <span class="preview-item">
              <strong>#2</strong> Samsung Galaxy Watch 7
            </span>
          </div>
          <a href="/best-smartwatches-2025/" class="btn btn-outline">
            View Guide
          </a>
        </div>
      </article>
      
      <!-- More guides... -->
    </div>
  </div>
</section>
```

---

##  IMPLEMENTATION ROADMAP

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 1.0

### Phase 1: Foundation (Week 1)
- [ ] Optimize existing dark mode CSS variables
- [ ] Install Space Grotesk + IBM Plex Mono fonts
- [ ] Create custom OG image (1200x630)
- [ ] Set up NewsArticle schema in RankMath
- [ ] Configure author schema for Alex Clearfield

### Phase 2: Navigation & Structure (Week 2)
- [ ] Build mega menu with MegaMenu Pro
- [ ] Create category landing pages
- [ ] Set up tag archive templates
- [ ] Implement breadcrumbs
- [ ] Add search functionality

### Phase 3: Homepage (Week 3)
- [ ] Design breaking news ticker
- [ ] Build featured story hero
- [ ] Create news grid layout
- [ ] Add "AI Explained" section
- [ ] Implement newsletter CTA

### Phase 4: Article Experience (Week 4)
- [ ] Design article template
- [ ] Add reading progress bar
- [ ] Create AI term tooltip system
- [ ] Build related posts component
- [ ] Add article share functionality

### Phase 5: Content & SEO (Week 5)
- [ ] Create AI 101 pillar page
- [ ] Build glossary with internal linking
- [ ] Optimize existing content
- [ ] Set up content categories
- [ ] Configure news sitemap

### Phase 6: Conversion (Week 6)
- [ ] Set up newsletter forms
- [ ] Create exit intent popup
- [ ] Add floating newsletter widget
- [ ] Configure email sequences
- [ ] A/B test CTA placements

---

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 1.0

### Phase 1: Foundation (Week 1)
- [ ] Update RankMath homepage meta (description: 120-160 chars)
- [ ] Create/upload custom OG image (1200x630px)
- [ ] Set up author persona (remove email display)
- [ ] Install Plus Jakarta Sans + Source Sans 3 fonts
- [ ] Configure Blocksy color palette
- [ ] Create logo variations (dark/light/icon)

### Phase 2: Structure (Week 2)
- [ ] Install and configure MegaMenu Pro
- [ ] Build navigation structure per architecture
- [ ] Create footer menus and widgets
- [ ] Set up 301 redirects if URLs change
- [ ] Configure breadcrumbs

### Phase 3: Homepage (Week 3)
- [ ] Design hero section in Elementor
- [ ] Build ecosystem selection cards
- [ ] Create resources showcase section
- [ ] Add blog/latest posts section
- [ ] Implement newsletter signup
- [ ] Add social proof elements

### Phase 4: Templates (Week 4)
- [ ] Create pillar page template
- [ ] Create blog post template
- [ ] Create comparison page template
- [ ] Create tool/calculator template
- [ ] Set up archive pages

### Phase 5: Content Enhancement (Week 5)
- [ ] Optimize existing pillar pages
- [ ] Add internal linking structure
- [ ] Create missing cluster content
- [ ] Add schema markup to all pages
- [ ] Optimize images (WebP, lazy load)

### Phase 6: Conversion (Week 6)
- [ ] Set up email capture forms
- [ ] Configure exit intent popup
- [ ] Add content upgrade CTAs
- [ ] A/B test hero CTAs
- [ ] Set up conversion tracking

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.8

### Phase 1: Critical Fixes (Week 1)
- [ ] Update homepage meta description (CRITICAL)
- [ ] Create custom OG image (replace Unsplash)
- [ ] Set up author persona (replace email display)
- [ ] Install Lexend + Source Sans 3 fonts
- [ ] Configure Blocksy colors

### Phase 2: Navigation & Structure (Week 2)
- [ ] Configure MegaMenu Pro with category structure
- [ ] Create category landing pages
- [ ] Set up breadcrumbs
- [ ] Build "How We Test" page
- [ ] Create editorial policy page

### Phase 3: Homepage Redesign (Week 3)
- [ ] Build hero section with search
- [ ] Create category browser grid
- [ ] Design Editor's Choice showcase
- [ ] Add latest reviews grid
- [ ] Build lab/trust section

### Phase 4: Review Template (Week 4)
- [ ] Design score summary card
- [ ] Build pros/cons component
- [ ] Create comparison table template
- [ ] Add FAQ schema blocks
- [ ] Build "Where to Buy" section

### Phase 5: Affiliate Optimization (Week 5)
- [ ] Configure RankMath review schema
- [ ] Set up affiliate link management
- [ ] Create deal alert system
- [ ] Build price comparison widgets
- [ ] A/B test CTA placements

### Phase 6: Content & Launch (Week 6)
- [ ] Create 3 pillar buying guides
- [ ] Optimize 10 top reviews with new template
- [ ] Set up newsletter capture
- [ ] Configure analytics events
- [ ] Launch and monitor

---

## ️ BLOCKSY CUSTOMIZER SETTINGS

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.4

### Global Settings

```json
{
  "colors": {
    "colorPalette": {
      "color1": "#2563eb",
      "color2": "#f59e0b",
      "color3": "#10b981",
      "color4": "#0f172a",
      "color5": "#f1f5f9"
    },
    "fontColor": "#1e293b",
    "linkColor": "#2563eb",
    "linkHoverColor": "#1d4ed8",
    "selectionBackground": "#2563eb",
    "selectionColor": "#ffffff"
  },
  "typography": {
    "rootTypography": {
      "family": "Source Sans 3",
      "variation": "n4",
      "size": "17px",
      "line-height": "1.7",
      "letter-spacing": "0"
    },
    "h1": {
      "family": "Plus Jakarta Sans",
      "variation": "n8",
      "size": "48px",
      "line-height": "1.1",
      "letter-spacing": "-0.02em"
    },
    "h2": {
      "family": "Plus Jakarta Sans",
      "variation": "n7",
      "size": "36px",
      "line-height": "1.2"
    },
    "h3": {
      "family": "Plus Jakarta Sans",
      "variation": "n6",
      "size": "28px",
      "line-height": "1.3"
    }
  },
  "layout": {
    "containerWidth": "1280px",
    "contentAreaSpacing": "80px",
    "narrowContainerWidth": "750px"
  },
  "buttons": {
    "buttonMinHeight": "48px",
    "buttonBorderRadius": "8px",
    "buttonPadding": "12px 28px",
    "buttonTextTransform": "none",
    "buttonFontWeight": "600"
  },
  "forms": {
    "formFieldBorderRadius": "8px",
    "formFieldHeight": "50px",
    "formFieldBorder": "1px solid #e2e8f0"
  }
}
```

### Header Configuration

```json
{
  "header": {
    "headerType": "type-1",
    "headerHeight": "80px",
    "headerBackground": "#ffffff",
    "stickyHeader": "yes",
    "stickyHeaderEffect": "slide",
    "stickyHeaderBackground": "rgba(255,255,255,0.98)",
    "stickyHeaderShadow": "0 2px 10px rgba(0,0,0,0.08)",
    "headerElements": {
      "row1": ["logo", "menu", "search", "cta-button"],
      "mobileRow": ["logo", "mobile-menu-trigger"]
    },
    "logo": {
      "type": "image",
      "maxHeight": "50px",
      "mobileMaxHeight": "40px"
    },
    "primaryMenu": {
      "menuItemsSpacing": "28px",
      "dropdownItemsSpacing": "0px",
      "dropdownTopOffset": "15px",
      "dropdownBoxShadow": "0 10px 40px rgba(0,0,0,0.15)"
    },
    "ctaButton": {
      "text": "Get Started",
      "link": "/getting-started/",
      "style": "primary",
      "size": "medium"
    }
  }
}
```

### Footer Configuration

```json
{
  "footer": {
    "footerType": "type-1",
    "footerBackground": "#0f172a",
    "footerTextColor": "#94a3b8",
    "footerLinkColor": "#f1f5f9",
    "footerLinkHoverColor": "#2563eb",
    "widgetColumns": 4,
    "widgetAreas": {
      "column1": {
        "title": "The Connected Haven",
        "content": "logo + tagline + social links"
      },
      "column2": {
        "title": "Ecosystems",
        "menu": "footer-ecosystems"
      },
      "column3": {
        "title": "Resources",
        "menu": "footer-resources"
      },
      "column4": {
        "title": "Newsletter",
        "content": "email signup form"
      }
    },
    "copyrightBar": {
      "background": "#0a0f1a",
      "text": "© 2025 The Connected Haven. All rights reserved.",
      "links": ["Privacy Policy", "Terms of Service", "Affiliate Disclosure"]
    }
  }
}
```

---

## General

### - Maintain "tech curator" voice consistently
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "tech curator" voice consistently
- Include relevant affiliate links (aidiscoverydigest-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Information overload, hype without context
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "forward analyst" voice consistently
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "forward analyst" voice consistently
- Include relevant affiliate links (aiinactionhub-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Hype without substance, doom-mongering
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "creative organizer" voice consistently
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "creative organizer" voice consistently
- Include relevant affiliate links (bulletjournals01-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Perfectionism pressure, gatekeeping
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "festive planner" voice consistently
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "festive planner" voice consistently
- Include relevant affiliate links (celebrationseason-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Excluding cultures, commercialism over meaning
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "nurturing guide" voice consistently
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "nurturing guide" voice consistently
- Include relevant affiliate links (familyflourish-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Judgment, one-size-fits-all, mom-shaming
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "empowering guide" voice consistently
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "empowering guide" voice consistently
- Include relevant affiliate links (manifestandalign-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Toxic positivity, bypassing real issues
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "scholarly wonder" voice consistently
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "scholarly wonder" voice consistently
- Include relevant affiliate links (mythicalarchives-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Oversimplification, cultural insensitivity
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "fitness enthusiast" voice consistently
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "fitness enthusiast" voice consistently
- Include relevant affiliate links (pulsegearreviews-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Unrealistic fitness claims
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "tech authority" voice consistently
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "tech authority" voice consistently
- Include relevant affiliate links (smarthomewizards-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Jargon overload, brand bias, outdated info
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "entrepreneurial strategist" voice consistently
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "entrepreneurial strategist" voice consistently
- Include relevant affiliate links (wealthfromai-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Get-rich-quick promises, unrealistic expectations
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### - Maintain "mystical warmth" voice consistently
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 1.0

### DO 
- Maintain "mystical warmth" voice consistently
- Include relevant affiliate links (witchcraftforbeginners-20)
- Optimize all images before upload
- Use proper heading hierarchy (H1 > H2 > H3)
- Include at least one CTA per post
- Add alt text to all images
- Follow E-E-A-T guidelines

### DON'T 
- Gatekeeping, cultural appropriation, fear-mongering
- Use generic AI-sounding phrases
- Publish without SEO optimization
- Forget internal links (minimum 3 per post)
- Skip featured images
- Ignore mobile responsiveness

---

### Content Generation
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for aidiscoverydigest.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using aidiscoverydigest-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for aiinactionhub.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using aiinactionhub-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for bulletjournals.net"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using bulletjournals01-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for celebrationseason.net"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using celebrationseason-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for family-flourish.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using familyflourish-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for manifestandalign.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using manifestandalign-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for mythicalarchives.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using mythicalarchives-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for pulsegearreviews.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using pulsegearreviews-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for smarthomewizards.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using smarthomewizards-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for wealthfromai.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using wealthfromai-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Generation
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.8

```
"Generate a pillar article about [TOPIC] for witchcraftforbeginners.com"
"Create 5 cluster articles supporting [PILLAR]"
"Write a product review for [PRODUCT] with affiliate links using witchcraftforbeginners-20"
```

### WordPress Operations
```
"Create a new post titled [TITLE] with category [CAT]"
"Update the homepage hero section"
"Add schema markup to existing posts"
```

### Automation
```
"Set up n8n workflow for daily content publishing"
"Configure Systeme.io blog sync"
"Create email welcome sequence"
```

### SEO
```
"Audit SEO for last 10 posts"
"Generate internal linking suggestions"
"Create FAQ schema for [POST]"
```

---

### Content Types
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.6

| Type | Word Count | Purpose |
|------|------------|---------|
| Pillar | 3,000-5,000 | Comprehensive guides, topical authority |
| Cluster | 1,500-2,500 | Supporting articles, internal links |
| Quick | 500-1,000 | News, tips, product spotlights |

### SEO Strategy
```yaml
focus: Topical authority building
approach: Hub and spoke content clusters
e_e_a_t_signals:
  - Author bios with credentials
  - External expert citations
  - First-hand experience mentions
  - Updated dates on all content
  
schema_types:
  - Article
  - HowTo
  - FAQPage
  - Product (for reviews)
  - Review
  
internal_linking:
  - Minimum 3 internal links per post
  - Link to pillar content from clusters
  - Use descriptive anchor text
```

### Publishing Schedule
```yaml
frequency: 3-5 posts per week (automated)
best_days: Tuesday, Wednesday, Thursday
best_times: 9am, 12pm, 3pm EST
evergreen_ratio: 80% evergreen, 20% timely
```

---
