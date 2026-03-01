# Design & UX Patterns

> 9 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## Tech Stack

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.4

- **Backend:** Python 3.11 + FastAPI
- **Frontend:** Jinja2 + Tailwind CSS + htmx
- **Charts:** Chart.js
- **HTTP Client:** httpx (async)

---

##  DARK MODE IMPLEMENTATION

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 1.0

### CSS Variables Switch

```css
/* Dark Mode (default for this site) */
:root,
[data-theme="dark"],
.clearainews-dark {
  --bg-primary: var(--clear-dark-bg);
  --bg-surface: var(--clear-dark-surface);
  --bg-elevated: var(--clear-dark-elevated);
  --border-color: var(--clear-dark-border);
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  
  color-scheme: dark;
}

/* Light Mode */
[data-theme="light"] {
  --bg-primary: var(--clear-light-bg);
  --bg-surface: var(--clear-light-surface);
  --bg-elevated: #ffffff;
  --border-color: #e2e8f0;
  --text-primary: #0f172a;
  --text-secondary: #334155;
  --text-muted: #64748b;
  
  color-scheme: light;
}

/* Component Styles */
body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  transition: background-color 0.3s ease, color 0.3s ease;
}

.news-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
}

.news-card:hover {
  background: var(--bg-elevated);
  border-color: var(--clear-primary);
}
```

### Theme Toggle Component

```html
<!-- Theme Switcher (in header) -->
<button class="theme-toggle" aria-label="Toggle theme">
  <svg class="theme-toggle__sun" viewBox="0 0 24 24">
    <!-- sun icon -->
  </svg>
  <svg class="theme-toggle__moon" viewBox="0 0 24 24">
    <!-- moon icon -->
  </svg>
</button>
```

```javascript
// Theme Toggle Script
const toggle = document.querySelector('.theme-toggle');
const html = document.documentElement;

toggle.addEventListener('click', () => {
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('clear-theme', next);
});

// Initialize from storage
const stored = localStorage.getItem('clear-theme');
if (stored) {
  html.setAttribute('data-theme', stored);
}
```

---

##  Content Templates

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.4

### Sabbat Kit Page Structure
1. **Cover** - Title, subtitle, main image, brand
2. **About [Sabbat]** - History, meaning, timing
3. **Correspondences** - Colors, crystals, herbs, symbols, foods
4. **Altar Setup** - Elements, layout, image reference
5. **Deity Devotional** - Invocation, prayer, offerings
6. **Main Ritual** - Step-by-step ceremony
7. **Spells & Magic** - 2-3 themed spells
8. **Kitchen Witchery** - Recipe + magical tea
9. **Journal Prompts** - 5 reflection questions
10. **Intentions** - Goal-setting worksheet
11. **Gratitude** - Thanksgiving practice
12. **Notes** - Blank journaling space

---

##  RESPONSIVE BREAKPOINTS

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 1.0

```css
/* Mobile First Approach */

/* Base: Mobile (0 - 639px) */
.container { padding: 0 16px; }
.haven-hero__title { font-size: var(--text-3xl); }
.ecosystem-grid { grid-template-columns: 1fr; gap: 24px; }

/* Tablet (640px - 1023px) */
@media (min-width: 640px) {
  .container { padding: 0 24px; }
  .haven-hero__title { font-size: var(--text-4xl); }
  .ecosystem-grid { grid-template-columns: repeat(2, 1fr); }
}

/* Desktop (1024px - 1279px) */
@media (min-width: 1024px) {
  .container { max-width: 1024px; margin: 0 auto; }
  .ecosystem-grid { grid-template-columns: repeat(4, 1fr); }
  .haven-hero { min-height: 90vh; }
}

/* Large Desktop (1280px+) */
@media (min-width: 1280px) {
  .container { max-width: 1280px; }
  .haven-hero__title { font-size: var(--text-5xl); }
}
```

---

##  COMPARISON TOOL

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

### Interactive Comparison Page

```html
<!-- Comparison Tool -->
<div class="wgr-compare-tool">
  <div class="container">
    <header class="compare-header">
      <h1>Compare Wearables</h1>
      <p>Select up to 4 devices to compare side-by-side</p>
    </header>
    
    <!-- Device Selector -->
    <div class="compare-selector">
      <div class="selector-slot" data-slot="1">
        <div class="selector-dropdown">
          <input type="text" placeholder="Search for a device...">
          <div class="dropdown-results">
            <!-- Dynamic results -->
          </div>
        </div>
      </div>
      <span class="vs-badge">VS</span>
      <div class="selector-slot" data-slot="2">
        <!-- Same structure -->
      </div>
      <button class="add-device-btn">+ Add Device</button>
    </div>
    
    <!-- Comparison Table -->
    <div class="compare-table">
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th class="device-col">
              <img src="..." alt="">
              <span>Apple Watch 10</span>
              <span class="score">9.2</span>
            </th>
            <th class="device-col">
              <img src="..." alt="">
              <span>Galaxy Watch 7</span>
              <span class="score">8.8</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr class="section-header">
            <td colspan="3">Display</td>
          </tr>
          <tr>
            <td>Screen Size</td>
            <td>1.96"</td>
            <td>1.47"</td>
          </tr>
          <tr>
            <td>Resolution</td>
            <td>502 x 410</td>
            <td>450 x 450</td>
          </tr>
          <!-- ... more rows ... -->
        </tbody>
      </table>
    </div>
    
    <!-- Winner Summary -->
    <div class="compare-summary">
      <h3>Our Recommendation</h3>
      <div class="summary-cards">
        <div class="summary-card">
          <h4>Best for iPhone Users</h4>
          <span class="winner">Apple Watch Series 10</span>
        </div>
        <div class="summary-card">
          <h4>Best Value</h4>
          <span class="winner">Samsung Galaxy Watch 7</span>
        </div>
      </div>
    </div>
  </div>
</div>
```

---

##  CUSTOM CSS SNIPPETS

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 1.0

```css
/* Haven Hero Animation */
.haven-hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  overflow: hidden;
}

.haven-hero__background video {
  position: absolute;
  top: 50%;
  left: 50%;
  min-width: 100%;
  min-height: 100%;
  transform: translate(-50%, -50%);
  object-fit: cover;
}

.haven-hero__overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    180deg,
    rgba(15, 23, 42, 0.85) 0%,
    rgba(15, 23, 42, 0.7) 50%,
    rgba(15, 23, 42, 0.9) 100%
  );
}

.gradient-text {
  background: var(--haven-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.pulse-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: var(--connected-green);
  border-radius: 50%;
  margin-right: 8px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.2); }
}

/* Ecosystem Cards */
.ecosystem-card {
  position: relative;
  padding: 32px;
  background: var(--haven-white);
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
  overflow: hidden;
}

.ecosystem-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: currentColor;
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.ecosystem-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.ecosystem-card:hover::before {
  transform: scaleX(1);
}

.ecosystem-card--alexa { color: var(--alexa-blue); }
.ecosystem-card--homekit { color: var(--homekit-orange); }
.ecosystem-card--google { color: var(--google-blue); }
.ecosystem-card--smartthings { color: var(--smartthings-green); }

/* Sticky TOC for Pillar Pages */
.haven-toc {
  position: sticky;
  top: 100px;
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  padding: 24px;
  background: var(--haven-light);
  border-radius: 12px;
}

.haven-toc__title {
  font-size: var(--text-sm);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--haven-muted);
  margin-bottom: 16px;
}

.haven-toc__list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.haven-toc__link {
  display: block;
  padding: 8px 0;
  color: var(--haven-text);
  text-decoration: none;
  border-left: 2px solid transparent;
  padding-left: 12px;
  margin-left: -12px;
  transition: all 0.2s ease;
}

.haven-toc__link:hover,
.haven-toc__link.active {
  color: var(--haven-primary);
  border-left-color: var(--haven-primary);
}
```

---

## ️ BLOCKSY CUSTOMIZER SETTINGS

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.8

```json
{
  "colors": {
    "colorPalette": {
      "color1": "#1e40af",
      "color2": "#ea580c",
      "color3": "#059669",
      "color4": "#111827",
      "color5": "#f3f4f6"
    }
  },
  "typography": {
    "rootTypography": {
      "family": "Source Sans 3",
      "variation": "n4",
      "size": "17px",
      "line-height": "1.7"
    },
    "h1": {
      "family": "Lexend",
      "variation": "n7",
      "size": "44px",
      "line-height": "1.1",
      "letter-spacing": "-0.02em"
    }
  },
  "header": {
    "stickyHeader": "yes",
    "headerBackground": "#ffffff",
    "ctaButton": {
      "text": "Start Here",
      "link": "/best-smart-home-devices/",
      "style": "primary"
    }
  }
}
```

---

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.6

### Global Configuration

```json
{
  "colors": {
    "colorPalette": {
      "color1": "#3b82f6",
      "color2": "#8b5cf6",
      "color3": "#22c55e",
      "color4": "#0a0a0f",
      "color5": "#f1f5f9"
    }
  },
  "darkMode": {
    "enabled": true,
    "defaultMode": "dark",
    "colorModeImplementation": "class",
    "toggleInHeader": true
  },
  "typography": {
    "rootTypography": {
      "family": "Inter",
      "variation": "n4",
      "size": "16px",
      "line-height": "1.7"
    },
    "h1": {
      "family": "Space Grotesk",
      "variation": "n7",
      "size": "52px",
      "line-height": "1.1",
      "letter-spacing": "-0.02em"
    }
  },
  "header": {
    "stickyHeader": "yes",
    "transparentHeader": "yes",
    "stickyHeaderBackground": "var(--bg-primary)"
  }
}
```

---

## General

### Design Philosophy
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.8

**"Modern Tech Picasso"** - Unexpected, striking, memorable designs that never look AI-generated.

### v0.dev Component Generation
```yaml
api_key: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
endpoint: https://api.v0.dev/v1/generate
use_for:
  - Hero sections
  - Feature cards
  - Pricing tables
  - Navigation components
  - Call-to-action blocks
```

---
