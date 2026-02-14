# Etsy POD Manager — OpenClaw Skill

## Purpose
Manage Etsy print-on-demand operations across witchcraft sub-niches. Generate designs via Midjourney/fal.ai, create Printify products, manage listings, track orders, and optimize for Etsy SEO.

## Trigger Phrases
- "create Etsy listing", "generate POD design", "Etsy sales report"
- "new [niche] design", "update listings", "bestseller analysis"
- "Printify sync", "order status", "Etsy SEO optimize"

## Sub-Niche Strategy

| Niche | Style | Colors | Target Audience |
|-------|-------|--------|-----------------|
| Cosmic Witch | Celestial, galaxy, astrology | Deep purples, midnight blue, gold | Astrology lovers, new witches |
| Cottage Witch | Cozy, botanical, herbs | Sage green, warm cream, brown | Nature lovers, kitchen witches |
| Green Witch | Plants, earth, forest | Forest green, earth tones | Herbalists, eco-conscious |
| Sea Witch | Ocean, shells, tides | Ocean blue, teal, pearl white | Beach lovers, water element |
| Moon Witch | Lunar phases, silver, night | Silver, black, pale blue | Moon followers, night owls |
| Crystal Witch | Gemstones, geometric, prisms | Amethyst purple, clear, rainbow | Crystal collectors |

## Product Types
- T-Shirts (Bella+Canvas 3001, Gildan 18000)
- Mugs (11oz, 15oz ceramic)
- Tote Bags (canvas)
- Stickers (vinyl, die-cut)
- Phone Cases
- Tapestries / Wall Art
- Journals / Notebooks
- Greeting Cards

## Design Generation Workflow
1. **Concept**: Define niche + product type + design theme
2. **Generate**: Create design via fal.ai (or queue Midjourney)
3. **Refine**: Upscale to print-ready resolution (300 DPI)
4. **Mockup**: Generate product mockups for listing
5. **List**: Create Printify product → sync to Etsy
6. **Optimize**: Write Etsy SEO title, tags, description

## Etsy SEO Template
```
Title: [Primary Keyword] | [Secondary Keyword] | [Style] | [Product Type] | [Gift Occasion]
Example: Cosmic Witch Shirt | Celestial Astrology Tee | Witchy Aesthetic | Mystical Gift for Her

Tags (13 max):
- cosmic witch, celestial shirt, astrology tee, witchy clothing
- mystical gift, witch aesthetic, pagan shirt, occult fashion
- moon stars shirt, witchcraft gift, spiritual tee, boho witch, gothic tee

Description:
[Hook — 1 line about the design]
[Product details — material, sizing, care]
[Gift angle — who it's perfect for]
[Shop pitch — browse more designs]
```

## Commands
```bash
# Generate a new design
etsy design --niche "cosmic witch" --product "t-shirt" --concept "moon phases with crystals"

# Create Printify product from design
etsy product --design ./designs/cosmic-moon-001.png --type "t-shirt" --title "Moon Phase Crystal Tee"

# Bulk list generation (5 products from one design across product types)
etsy bulk --design ./designs/cosmic-moon-001.png --products "tshirt,mug,tote,sticker,phonecase"

# Sales dashboard
etsy sales --period month

# Bestseller analysis
etsy bestsellers --top 20

# SEO optimization pass on all active listings
etsy seo-optimize --all

# Order fulfillment status
etsy orders --status pending
```

## Revenue Tracking
- Track daily/weekly/monthly sales per niche
- Calculate profit margins (Etsy fees + Printify cost + Etsy ads)
- Identify best-selling designs for replication
- Alert on trending search terms in witchcraft POD space
