---

## 🎯 EXECUTIVE SUMMARY

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

## 🎨 DESIGN SYSTEM ENHANCEMENT

### Brand Identity

```
BRAND NAME: Wearable Gear Reviews
TAGLINE: "Your Daily Pulse on Wearable Tech"
SECONDARY: "Data-driven reviews that cut through the hype"

BRAND ESSENCE: The fitness-first tech reviewer who tests products in real workouts

BRAND PERSONALITY:
- Data-Driven Tester (not spec sheet reader)
- Fitness Enthusiast (not desk reviewer)
- Hype Cutter (not marketing echo)
- Lifestyle Matcher (not one-size-fits-all)

REVIEWER PERSONA: The WGR Team / Lead: "Alex Gear" or similar
- Active lifestyle focus
- Tests products during actual workouts
- Compares with competitors head-to-head
- Uses devices for 2+ weeks minimum
```

### Color Palette

```css
:root {
  /* Primary - Performance Orange (energy, action, fitness) */
  --wgr-primary: #f97316;
  --wgr-primary-dark: #ea580c;
  --wgr-primary-light: #fb923c;
  
  /* Secondary - Tech Blue (precision, technology, trust) */
  --wgr-secondary: #0ea5e9;
  --wgr-secondary-dark: #0284c7;
  --wgr-secondary-light: #38bdf8;
  
  /* Accent - Health Green (vitality, achievement, tracking) */
  --wgr-accent: #22c55e;
  --wgr-accent-dark: #16a34a;
  --wgr-accent-light: #4ade80;
  
  /* Neutrals - Modern Dark */
  --wgr-dark: #0f172a;
  --wgr-darker: #020617;
  --wgr-text: #1e293b;
  --wgr-muted: #64748b;
  --wgr-light: #f1f5f9;
  --wgr-white: #ffffff;
  
  /* Rating Colors */
  --rating-excellent: #22c55e;  /* 9-10 */
  --rating-great: #84cc16;      /* 8-8.9 */
  --rating-good: #eab308;       /* 7-7.9 */
  --rating-average: #f97316;    /* 6-6.9 */
  --rating-poor: #ef4444;       /* <6 */
  
  /* Category Colors */
  --cat-smartwatch: #8b5cf6;
  --cat-fitness: #22c55e;
  --cat-health: #ef4444;
  --cat-audio: #3b82f6;
  --cat-sports: #f97316;
  
  /* Gradients */
  --wgr-gradient: linear-gradient(135deg, var(--wgr-primary) 0%, var(--wgr-secondary) 100%);
  --hero-gradient: linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(15,23,42,0.8) 100%);
  --pulse-gradient: radial-gradient(circle, var(--wgr-primary)30 0%, transparent 70%);
}
```

### Typography System

```css
:root {
  /* Display - Athletic, bold headers */
  --font-display: 'Outfit', 'DM Sans', system-ui, sans-serif;
  
  /* Body - Clean, readable reviews */
  --font-body: 'Nunito Sans', 'Open Sans', system-ui, sans-serif;
  
  /* Mono - Specs and data */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  
  /* Scale */
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 2rem;
  --text-4xl: 2.5rem;
  --text-5xl: 3.5rem;
}

/* Review Typography */
.wgr-review h1 {
  font-family: var(--font-display);
  font-size: var(--text-4xl);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.02em;
}

.wgr-review .review-body {
  font-family: var(--font-body);
  font-size: var(--text-lg);
  line-height: 1.8;
}

.wgr-specs {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
}
```

---

## 🏗️ SITE ARCHITECTURE

### Navigation Structure

```
PRIMARY NAVIGATION:
├── Reviews ▼ (mega menu)
│   ├── 📱 Smartwatches
│   │   ├── Apple Watch
│   │   ├── Samsung Galaxy Watch
│   │   ├── Garmin
│   │   └── All Smartwatches
│   ├── 💪 Fitness Trackers
│   │   ├── Fitbit
│   │   ├── Whoop
│   │   ├── Oura Ring
│   │   └── All Fitness Trackers
│   ├── ❤️ Health Monitors
│   │   ├── CGMs (Glucose)
│   │   ├── Blood Pressure
│   │   ├── Sleep Trackers
│   │   └── All Health
│   ├── 🎧 Audio Wearables
│   │   ├── Earbuds
│   │   ├── Headphones
│   │   └── All Audio
│   └── 🏃 Sports Wearables
│       ├── Running Watches
│       ├── Cycling Computers
│       └── All Sports
│
├── Best Of ▼
│   ├── Best Smartwatches 2025
│   ├── Best Fitness Trackers 2025
│   ├── Best for Runners
│   ├── Best for Sleep
│   ├── Best Budget Picks
│   └── All Buying Guides
│
├── Compare
│   └── (Interactive comparison tool)
│
├── Guides ▼
│   ├── Buying Guides
│   ├── How-To Guides
│   ├── Tech Explained
│   └── Fitness Tips
│
└── About
    ├── How We Test
    ├── About WGR
    └── Contact
```

### Category Structure

```
PRODUCT CATEGORIES:
├── smartwatches/
│   ├── apple-watch/
│   ├── samsung-galaxy-watch/
│   ├── garmin/
│   └── google-pixel-watch/
├── fitness-trackers/
│   ├── fitbit/
│   ├── whoop/
│   ├── oura/
│   └── xiaomi/
├── health-monitors/
├── audio-wearables/
└── sports-wearables/

CONTENT TYPES:
├── reviews/ (individual product reviews)
├── comparisons/ (X vs Y articles)
├── best/ (roundup buying guides)
├── guides/ (how-to, buying guides)
└── news/ (product launches, updates)
```

---

## 📝 SEO IMPLEMENTATION

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

## 🖥️ HOMEPAGE DESIGN

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

## 📱 REVIEW PAGE TEMPLATE

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
          <h4>👍 What We Loved</h4>
          <ul>
            <li>Gorgeous larger display</li>
            <li>Incredibly thin design</li>
            <li>Excellent fitness tracking</li>
            <li>Bright always-on mode</li>
          </ul>
        </div>
        <div class="cons">
          <h4>👎 What Could Be Better</h4>
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

## 🔄 COMPARISON TOOL

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

## 💰 AFFILIATE INTEGRATION

### Content Egg Configuration

```php
// Content Egg Product Display
// Shortcode usage in reviews

// Single product box
[content-egg module=Amazon template=custom/product-box]

// Price comparison
[content-egg module=Amazon,Ebay template=price-comparison]

// Product gallery
[content-egg module=Amazon template=grid limit=6]
```

### Custom Affiliate Block

```html
<!-- Where to Buy Box -->
<div class="wgr-buy-box">
  <h4>Where to Buy</h4>
  <div class="buy-options">
    <a href="#" class="buy-option" rel="sponsored nofollow">
      <img src="/retailers/amazon.svg" alt="Amazon">
      <span class="retailer">Amazon</span>
      <span class="price">$399</span>
      <span class="btn">Check Price</span>
    </a>
    <a href="#" class="buy-option" rel="sponsored nofollow">
      <img src="/retailers/bestbuy.svg" alt="Best Buy">
      <span class="retailer">Best Buy</span>
      <span class="price">$399</span>
      <span class="btn">Check Price</span>
    </a>
    <a href="#" class="buy-option" rel="sponsored nofollow">
      <img src="/retailers/apple.svg" alt="Apple">
      <span class="retailer">Apple.com</span>
      <span class="price">$399</span>
      <span class="btn">Check Price</span>
    </a>
  </div>
  <p class="affiliate-disclosure">
    * Prices updated hourly. We earn commission on qualifying purchases.
  </p>
</div>
```

---

## 🚀 IMPLEMENTATION ROADMAP

### Phase 1: Design System Polish (Week 1)
- [ ] Audit existing wgr-* CSS classes
- [ ] Implement enhanced color variables
- [ ] Install Outfit + Nunito Sans fonts
- [ ] Create consistent component styles
- [ ] Build reusable review card component

### Phase 2: Review Template (Week 2)
- [ ] Design score card component
- [ ] Build pros/cons block
- [ ] Create specs table template
- [ ] Implement review gallery
- [ ] Add reading progress indicator

### Phase 3: Comparison Tool (Week 3)
- [ ] Build product selector component
- [ ] Create comparison table layout
- [ ] Implement JavaScript comparison logic
- [ ] Add winner summary section
- [ ] Create shareable comparison URLs

### Phase 4: Category & Archive Pages (Week 4)
- [ ] Design category landing pages
- [ ] Build filtering system
- [ ] Create brand archive pages
- [ ] Implement sorting options
- [ ] Add "Best Of" badges

### Phase 5: Affiliate Optimization (Week 5)
- [ ] Configure Content Egg displays
- [ ] Create custom "Where to Buy" block
- [ ] Add price tracking widgets
- [ ] Implement deal alerts section
- [ ] A/B test CTA placements

### Phase 6: Conversion & Speed (Week 6)
- [ ] Optimize images (WebP, lazy load)
- [ ] Implement critical CSS
- [ ] Add newsletter popups
- [ ] Create exit intent offers
- [ ] Set up conversion tracking

---

## 📁 FILE STRUCTURE

```
wearablegearreviews.com/
├── wp-content/
│   ├── themes/
│   │   └── wgr-design-system/ (existing custom theme)
│   │       ├── style.css
│   │       ├── functions.php
│   │       ├── template-parts/
│   │       │   ├── review-card.php
│   │       │   ├── score-card.php
│   │       │   ├── pros-cons.php
│   │       │   └── buy-box.php
│   │       └── assets/
│   │           ├── css/
│   │           │   ├── wgr-design-system.css
│   │           │   ├── review-page.css
│   │           │   ├── comparison-tool.css
│   │           │   └── components.css
│   │           ├── js/
│   │           │   ├── comparison-tool.js
│   │           │   ├── score-animation.js
│   │           │   └── affiliate-tracking.js
│   │           └── images/
│   │               ├── brands/
│   │               ├── retailers/
│   │               └── icons/
│   └── plugins/
│       └── content-egg-templates/
│           ├── product-box.php
│           └── price-comparison.php
```

---

## 📈 SUCCESS METRICS

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

## 🔧 CUSTOM CSS

```css
/* WGR Design System Enhancements */

/* Score Circle Animation */
.score-circle {
  position: relative;
  width: 120px;
  height: 120px;
}

.score-ring {
  position: absolute;
  transform: rotate(-90deg);
}

.score-ring circle {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
}

.score-ring .bg {
  stroke: var(--wgr-light);
}

.score-ring .progress {
  stroke: var(--wgr-gradient);
  stroke-dasharray: 339.292;
  stroke-dashoffset: calc(339.292 * (1 - var(--score) / 10));
  transition: stroke-dashoffset 1s ease-out;
}

.score-value {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 800;
}

/* Score Bar Animation */
.score-bar {
  height: 8px;
  background: var(--wgr-light);
  border-radius: 4px;
  overflow: hidden;
  flex: 1;
  margin: 0 12px;
}

.score-bar::after {
  content: '';
  display: block;
  height: 100%;
  width: calc(var(--score) * 10%);
  background: var(--wgr-gradient);
  border-radius: 4px;
  transition: width 0.8s ease-out;
}

/* Review Card Hover */
.review-card {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: var(--wgr-white);
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  transition: all 0.3s ease;
}

.review-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}

.review-card__rating {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 48px;
  height: 48px;
  background: var(--wgr-gradient);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: var(--text-lg);
}

/* Pros/Cons Grid */
.pros-cons-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.pros, .cons {
  padding: 24px;
  border-radius: 12px;
}

.pros {
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.cons {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.pros h4, .cons h4 {
  margin-bottom: 16px;
  font-size: var(--text-lg);
}

.pros li::marker { color: var(--rating-excellent); }
.cons li::marker { color: var(--rating-poor); }

/* Buy Box */
.wgr-buy-box {
  background: var(--wgr-light);
  border-radius: 16px;
  padding: 24px;
  margin: 32px 0;
}

.buy-option {
  display: flex;
  align-items: center;
  padding: 16px;
  background: white;
  border-radius: 12px;
  margin-bottom: 12px;
  transition: all 0.2s ease;
}

.buy-option:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.buy-option .btn {
  margin-left: auto;
  background: var(--wgr-primary);
  color: white;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 600;
}

/* Category Cards */
.category-card {
  position: relative;
  padding: 32px;
  border-radius: 20px;
  background: white;
  border: 2px solid transparent;
  transition: all 0.3s ease;
  overflow: hidden;
}

.category-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: currentColor;
}

.category-card:hover {
  border-color: currentColor;
  transform: translateY(-4px);
}

.category-card--smartwatch { color: var(--cat-smartwatch); }
.category-card--fitness { color: var(--cat-fitness); }
.category-card--health { color: var(--cat-health); }
.category-card--audio { color: var(--cat-audio); }
.category-card--sports { color: var(--cat-sports); }
```

---

**END OF BLUEPRINT**

*Wearable Gear Reviews transformation guide. Build the definitive wearable tech review destination.*

**Document Version:** 1.0
**Created:** 2025-12-16
**Author:** Claude (AI Publishing Empire Assistant)

---
