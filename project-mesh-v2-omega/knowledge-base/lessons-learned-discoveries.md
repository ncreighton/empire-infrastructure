# Lessons Learned & Discoveries

> 6 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## Important Notes

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

- ZimmWriter must be RUNNING and VISIBLE for the controller to work
- Run `discover_controls.py` after any ZimmWriter update to re-map control names
- Run `discover_all_screens.py --list-dropdown-values` for exhaustive screen mapping
- The API server must run ON THE SAME Windows machine as ZimmWriter
- File paths in CSV loading must use absolute Windows paths (C:\\...)
- Feature toggle buttons show "Enabled"/"Disabled" in their button text
- Profile saving uses Update Profile (cid=31), NOT Save Profile (cid=30)

---

## Notes

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.4

- This project emphasizes accessibility and inclusivity
- All rituals must have variations for different abilities/budgets
- Never recommend practices that could cause harm
- Garden sage preferred over white sage
- Dark theme is core to brand identity
- Moon phase calculations are astronomical, not astrological

---

## RATE LIMITS

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.4

- Max 100 requests per hour
- Random delays 2-6 seconds
- Residential proxy rotation
- Human-like behavior patterns

If blocked: Wait 30 min, retry with different proxy

---

## Shared Systems (Current Versions)

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 0.4

| System | Version | Criticality | Usage |
|--------|---------|-------------|-------|
| api-retry | 1.0.0 [OK] | high | hourly |

---

##  UNIQUE FEATURES

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.4

### AI Term Tooltips

```css
/* Inline AI terminology tooltips */
.ai-term {
  border-bottom: 1px dashed var(--clear-primary);
  cursor: help;
  position: relative;
}

.ai-term::after {
  content: attr(data-definition);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-elevated);
  border: 1px solid var(--border-color);
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.875rem;
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  transition: all 0.2s ease;
  z-index: 100;
}

.ai-term:hover::after {
  opacity: 1;
  visibility: visible;
  bottom: calc(100% + 8px);
}
```

### Breaking News Ticker

```javascript
// Auto-refreshing breaking news ticker
class NewsTicker {
  constructor() {
    this.ticker = document.querySelector('.ticker-content');
    this.refreshInterval = 60000; // 1 minute
    this.init();
  }
  
  async init() {
    await this.fetchBreaking();
    setInterval(() => this.fetchBreaking(), this.refreshInterval);
  }
  
  async fetchBreaking() {
    const response = await fetch('/wp-json/wp/v2/posts?categories=breaking&per_page=5');
    const posts = await response.json();
    this.renderTicker(posts);
  }
  
  renderTicker(posts) {
    const html = posts.map(p => `
      <a href="${p.link}" class="ticker-item">
        <span class="ticker-time">${this.timeAgo(p.date)}</span>
        ${p.title.rendered}
      </a>
    `).join('<span class="ticker-separator">-</span>');
    this.ticker.innerHTML = html;
  }
}
```

### Reading Progress Bar

```css
/* Article reading progress indicator */
.reading-progress {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--bg-surface);
  z-index: 9999;
}

.reading-progress__bar {
  height: 100%;
  background: var(--neural-gradient);
  width: 0%;
  transition: width 0.1s ease;
}
```

---

##  Tips for Claude Code Usage

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.4

1. **Always generate images first** - They take time and you want to review before embedding
2. **Use HTML preview** - Check layout before PDF generation
3. **Batch similar products** - Generate all sabbat kits in one session
4. **Save image prompts** - Build a library of working prompts
5. **Version control** - Commit templates after major improvements
6. **Test print** - PDF screens differently than paper

---

*VelvetVeilPrintables © 2026 - All Rights Reserved*

---
