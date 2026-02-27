# Buy Me a Coffee — Substack Integration Strategy

> Ready-to-paste CTAs for your Substack newsletter (`witchcraftb.substack.com`).
> See `templates/substack_ctas.md` for the complete CTA template library.

---

## Strategy Overview

Every Substack post should include **one** BMC call-to-action. Rotate placement and style to avoid fatigue.

| Placement | Frequency | Best For |
|-----------|-----------|----------|
| Top banner (before content) | 1 in 4 posts | Major launches, new tier announcements |
| Mid-article callout | 1 in 2 posts | Contextual tie-ins (referencing exclusive content) |
| Bottom CTA | Every post | Default — lowest friction, highest conversion for engaged readers |

---

## Placement 1: Top Banner (Sparingly)

Use this only for launches, new products, or special announcements. Overuse trains readers to skip it.

```markdown
---
🌙 **NEW: The Sacred Circle is Open**

Witchcraft For Beginners now has a members-only community on Buy Me a Coffee — with exclusive spell kits, grimoire references, tarot pulls, and the complete VelvetVeil digital library.

**[Join the Circle →](https://buymeacoffee.com/witchcraftyou)**

---
```

---

## Placement 2: Mid-Article Callout

Insert naturally when the article topic connects to exclusive content. Examples:

**When discussing herbs:**
```markdown
> 🌿 **Members' Grimoire Reference:** The Candlelight Circle gets a monthly herb & crystal quick reference card — this month covers the 7 essential protective herbs with full correspondences. [See what's inside →](https://buymeacoffee.com/witchcraftyou)
```

**When discussing moon phases:**
```markdown
> 🌕 **Exclusive Moon Ritual:** Every full moon, members receive a complete step-by-step ritual with materials list, timing guide, and journal prompts. This month's Snow Moon cleansing ritual is available now. [Join the Candlelight Circle →](https://buymeacoffee.com/witchcraftyou)
```

**When discussing spells:**
```markdown
> ✨ **Moonlit Coven Exclusive:** Members receive a monthly spell kit PDF with complete workings — ingredients, instructions, and optimal timing. This month's kit is the Ostara Egg Blessing & Intention Planting ritual. [Explore membership →](https://buymeacoffee.com/witchcraftyou)
```

**When discussing digital products:**
```markdown
> 📚 **High Priestess Circle members** get instant access to the complete VelvetVeil digital grimoire library — 30+ PDFs, 800+ pages, a $390+ value included with membership. [Learn more →](https://buymeacoffee.com/witchcraftyou)
```

---

## Placement 3: Bottom CTA (Default)

Use one of these at the end of every post. Rotate for variety.

**Version A — Community Focus:**
```markdown
---

*This post was free for everyone. If it added something to your practice, consider joining the Sacred Circle — your support keeps the magick flowing.*

**[☕ Buy me a potion](https://buymeacoffee.com/witchcraftyou)** *or* **[🌙 Join the Circle](https://buymeacoffee.com/witchcraftyou)**

---
```

**Version B — Exclusive Content Tease:**
```markdown
---

*Want more? Members get exclusive spell kits, monthly tarot pulls, grimoire references, and early access to every post.*

**[Explore membership tiers →](https://buymeacoffee.com/witchcraftyou)**

---
```

**Version C — Gratitude Focus:**
```markdown
---

*Witchcraft For Beginners is supported by practitioners like you. Every potion fuels another guide, another ritual, another resource for our community.*

**[Support the craft 🌙](https://buymeacoffee.com/witchcraftyou)**

---
```

**Version D — Product Highlight:**
```markdown
---

*Prefer a one-time purchase? The BMC shop has grimoire PDFs, spell books, and ritual kits — including the BMC-exclusive Complete Digital Grimoire Library ($59.99 for 30+ PDFs).*

**[Browse the shop →](https://buymeacoffee.com/witchcraftyou)**

---
```

---

## Rotation Schedule

| Post # | Placement | CTA Version |
|--------|-----------|-------------|
| 1 | Top (launch) | Launch announcement |
| 2 | Bottom | Version A |
| 3 | Mid + Bottom | Contextual + Version B |
| 4 | Bottom | Version C |
| 5 | Mid + Bottom | Contextual + Version D |
| 6 | Bottom | Version A |
| 7 | Mid + Bottom | Contextual + Version B |
| 8 | Bottom | Version C |
| Then repeat... | | |

---

## Substack Button Configuration

Substack allows custom buttons. Add this to your Substack settings:

| Setting | Value |
|---------|-------|
| Subscribe CTA | "Join the free newsletter" |
| Custom button text | "Support on BMC" |
| Custom button URL | `https://buymeacoffee.com/witchcraftyou` |

---

## WordPress Blog Integration

See `templates/blog_sidebar_widget.html` for a brand-matched sidebar widget.

**Placement locations:**
1. Sidebar widget (persistent on all pages)
2. After-post CTA (single posts only)
3. Footer link (site-wide)

**WordPress functions.php snippet** (optional — adds BMC link to nav menu):
```php
// Add to your child theme's functions.php
function add_bmc_menu_item($items, $args) {
    if ($args->theme_location == 'primary') {
        $items .= '<li class="menu-item bmc-link"><a href="https://buymeacoffee.com/witchcraftyou" target="_blank" rel="noopener">Support Us ☕</a></li>';
    }
    return $items;
}
add_filter('wp_nav_menu_items', 'add_bmc_menu_item', 10, 2);
```

---

## Etsy Cross-Promotion

Add to your Etsy shop's About section or listing descriptions:

```
💜 Love our grimoire PDFs? Get the complete library (30+ PDFs) for just $59.99 with a High Priestess membership on Buy Me a Coffee — that's 85% off buying individually.

→ buymeacoffee.com/witchcraftyou
```

---

## Setup Checklist

- [ ] Add BMC CTA to the next Substack newsletter
- [ ] Configure Substack custom button
- [ ] Install WordPress sidebar widget
- [ ] Add footer link to WordPress site
- [ ] Update Etsy shop About section
- [ ] Schedule first rotation cycle (8 posts)
