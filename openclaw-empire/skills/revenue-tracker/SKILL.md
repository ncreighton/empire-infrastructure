# Revenue Tracker — OpenClaw Skill

## Purpose
Track revenue across all income streams: WordPress ad revenue, affiliate commissions, KDP royalties, Etsy sales, Substack subscriptions, and YouTube/podcast monetization. Surface insights, alert on anomalies, and project growth.

## Trigger Phrases
- "revenue report", "how much did I make", "income dashboard"
- "best performing site", "revenue by source", "monthly projections"
- "affiliate earnings", "KDP royalties", "Etsy profit"

## Revenue Streams

| Source | Sites/Products | Tracking Method |
|--------|---------------|-----------------|
| Display Ads (Mediavine/AdSense) | All 16 WordPress sites | Analytics API |
| Affiliate (Amazon, ShareASale, etc.) | All sites via Content Egg | Affiliate dashboard scrape |
| KDP Royalties | 20+ books | KDP Reports |
| Etsy POD Sales | 6 sub-niche shops | Etsy API |
| Substack Subscriptions | Witchcraft for Beginners | Substack dashboard |
| YouTube Ad Revenue | Multiple channels | YouTube Analytics API |
| Sponsored Content | Select sites | Manual tracking |
| Digital Products | Printables, templates | WooCommerce / Gumroad |

## Data Structure
```json
{
  "date": "2026-02-14",
  "streams": {
    "ads": {
      "witchcraft": 45.20,
      "smarthome": 32.10,
      "aiaction": 28.50,
      "total": 185.30
    },
    "affiliate": {
      "amazon": 120.00,
      "shareasale": 45.00,
      "total": 165.00
    },
    "kdp": {
      "ebooks": 35.00,
      "paperbacks": 22.00,
      "total": 57.00
    },
    "etsy": {
      "gross": 89.00,
      "fees": -18.50,
      "printify_cost": -35.00,
      "net": 35.50
    },
    "substack": 15.00,
    "youtube": 8.50,
    "grand_total": 466.30
  }
}
```

## Storage
```
~/.openclaw/workspace/revenue/
├── daily/
│   ├── 2026-02-14.json
│   └── ...
├── monthly/
│   ├── 2026-02.json
│   └── ...
├── annual/
│   └── 2026.json
└── config.json
```

## Commands
```bash
# Today's revenue
revenue today

# This week/month/year
revenue report --period week|month|year

# By source
revenue breakdown --source ads|affiliate|kdp|etsy|substack

# By site
revenue site --name witchcraft --period month

# Top performers
revenue top --metric revenue --count 5

# Growth comparison
revenue compare --period month --vs last-month

# Set revenue goals
revenue goal --monthly 5000

# Anomaly detection (unusual drops or spikes)
revenue alerts

# Export to Google Sheets
revenue export --format sheets --sheet-id XXXXX
```

## Alert Rules
- Daily revenue drops >30% from 7-day average → WhatsApp alert
- Any site earning $0 for 48+ hours → investigate
- Affiliate link broken (404) → immediate alert
- KDP book sales spike → analyze cause, replicate
- Monthly goal tracking: on/off pace notifications every Monday

## Projections
- Linear projection based on trailing 30-day average
- Seasonal adjustment (witchcraft peaks: Oct-Nov, Sabbats)
- Growth rate calculation per stream
- Break-even analysis for new initiatives
