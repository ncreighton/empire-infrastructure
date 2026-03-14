---

## 🎯 EXECUTIVE SUMMARY

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

## 🎨 DESIGN SYSTEM

### Brand Identity

```
BRAND NAME: Smart Home Gear Reviews
TAGLINE: "Lab-Tested Reviews You Can Trust"
SECONDARY: "Expert reviews. Rigorous testing. Honest verdicts."

BRAND ESSENCE: The Consumer Reports of smart home products

BRAND PERSONALITY:
- Rigorous Tester (not quick previewer)
- Data-Driven Expert (not opinion blogger)
- Consumer Advocate (not brand promoter)
- Practical Guide (not tech enthusiast)

REVIEWER PERSONA: "The SHGR Lab Team" / Lead: "Marcus Gear" or similar
- Former tech journalist background
- Tests products for 30+ days minimum
- Maintains testing methodology documentation
- Focuses on real-world usability
```

### Color Palette

```css
:root {
  /* Primary - Expert Blue (trust, authority, precision) */
  --shgr-primary: #1e40af;
  --shgr-primary-dark: #1e3a8a;
  --shgr-primary-light: #3b82f6;
  
  /* Secondary - Lab Orange (testing, verification, results) */
  --shgr-secondary: #ea580c;
  --shgr-secondary-dark: #c2410c;
  --shgr-secondary-light: #f97316;
  
  /* Accent - Trust Green (verified, approved, recommended) */
  --shgr-accent: #059669;
  --shgr-accent-dark: #047857;
  --shgr-accent-light: #10b981;
  
  /* Neutrals */
  --shgr-dark: #111827;
  --shgr-text: #1f2937;
  --shgr-muted: #6b7280;
  --shgr-light: #f3f4f6;
  --shgr-white: #ffffff;
  
  /* Rating System */
  --rating-10: #059669;     /* Outstanding */
  --rating-9: #10b981;      /* Excellent */
  --rating-8: #22c55e;      /* Great */
  --rating-7: #84cc16;      /* Good */
  --rating-6: #eab308;      /* Decent */
  --rating-5: #f97316;      /* Average */
  --rating-low: #ef4444;    /* Below Average */
  
  /* Badge Colors */
  --badge-editors-choice: #7c3aed;
  --badge-best-value: #059669;
  --badge-top-rated: #dc2626;
  --badge-lab-tested: #1e40af;
  
  /* Gradients */
  --shgr-gradient: linear-gradient(135deg, var(--shgr-primary) 0%, var(--shgr-primary-light) 100%);
  --hero-gradient: linear-gradient(180deg, rgba(17,24,39,0.9) 0%, rgba(17,24,39,0.7) 100%);
  --trust-gradient: linear-gradient(135deg, var(--shgr-primary) 0%, var(--shgr-accent) 100%);
}
```

### Typography System

```css
:root {
  /* Display - Authoritative, clear headers */
  --font-display: 'Lexend', 'DM Sans', system-ui, sans-serif;
  
  /* Body - Professional, readable */
  --font-body: 'Source Sans 3', 'Open Sans', system-ui, sans-serif;
  
  /* Mono - Specs and technical data */
  --font-mono: 'IBM Plex Mono', 'Fira Code', monospace;
  
  /* Scale */
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 2rem;
  --text-4xl: 2.75rem;
  --text-5xl: 3.5rem;
}

/* Authority Typography */
h1 {
  font-family: var(--font-display);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
}

.review-title {
  font-size: var(--text-4xl);
}

.section-title {
  font-size: var(--text-3xl);
  position: relative;
  padding-bottom: 16px;
}

.section-title::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 60px;
  height: 4px;
  background: var(--shgr-gradient);
  border-radius: 2px;
}
```

---

## 🏗️ SITE ARCHITECTURE

### Navigation Structure

```
PRIMARY NAVIGATION:
├── Reviews ▼ (mega menu)
│   ├── 💡 Smart Lighting
│   │   ├── Smart Bulbs
│   │   ├── Light Strips
│   │   ├── Smart Switches
│   │   └── All Lighting Reviews
│   ├── 🔒 Smart Security
│   │   ├── Security Cameras
│   │   ├── Video Doorbells
│   │   ├── Smart Locks
│   │   └── All Security Reviews
│   ├── 🌡️ Climate Control
│   │   ├── Smart Thermostats
│   │   ├── Smart Fans
│   │   ├── Air Quality Monitors
│   │   └── All Climate Reviews
│   ├── 🔊 Smart Speakers
│   │   ├── Amazon Echo
│   │   ├── Google Nest
│   │   ├── Apple HomePod
│   │   └── All Speaker Reviews
│   ├── 📺 Smart Displays
│   │   ├── Echo Show
│   │   ├── Nest Hub
│   │   └── All Display Reviews
│   └── 🏠 Smart Hubs
│       ├── Hub Reviews
│       ├── Protocol Guides
│       └── All Hub Reviews
│
├── Best Of ▼
│   ├── Best Smart Home Devices 2025
│   ├── Best Under $50
│   ├── Best for Beginners
│   ├── Best for Alexa
│   ├── Best for HomeKit
│   └── All Buying Guides
│
├── Lab ▼
│   ├── How We Test
│   ├── Testing Methodology
│   ├── Lab Equipment
│   └── Benchmark Results
│
├── Deals
│   └── (Current sales & discounts)
│
└── About
    ├── About SHGR
    ├── The Review Team
    ├── Editorial Policy
    └── Contact
```

### Testing Categories Framework

```
REVIEW SCORING CATEGORIES:
├── Setup & Installation (10%)
│   └── Ease of setup, app quality, documentation
├── Performance (25%)
│   └── Core function execution, reliability, speed
├── Features (20%)
│   └── Feature set, automation support, integrations
├── Build Quality (15%)
│   └── Materials, durability, design
├── App & Software (15%)
│   └── App design, updates, cloud vs local
├── Value (15%)
│   └── Price vs performance vs competition

BADGE SYSTEM:
- 🏆 Editor's Choice: Score 9.0+ with exceptional performance
- 💰 Best Value: Score 8.0+ with excellent price/performance
- ⭐ Top Rated: Highest in category
- 🔬 Lab Tested: All reviews (default badge)
- ⚠️ Caution: Known issues or concerns
```

---

## 📝 SEO IMPLEMENTATION

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

## 🖥️ HOMEPAGE DESIGN

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
      <span class="section-badge">🏆 Top Picks</span>
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
        <span class="section-badge">🔬 Our Process</span>
        <h2 class="section-title">The SHGR Testing Lab</h2>
        <p class="lab-showcase__intro">
          Every product goes through our rigorous multi-week testing protocol. 
          No quick impressions—real-world performance data.
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

## 📱 REVIEW PAGE TEMPLATE

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
          🏆 Editor's Choice
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

## 🛠️ BLOCKSY CUSTOMIZER SETTINGS

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

## 🚀 IMPLEMENTATION ROADMAP

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

## 📁 FILE STRUCTURE

```
smarthomegearreviews.com/
├── wp-content/
│   ├── themes/
│   │   └── blocksy-child/
│   │       ├── style.css
│   │       ├── functions.php
│   │       ├── template-parts/
│   │       │   ├── score-card.php
│   │       │   ├── pros-cons.php
│   │       │   ├── where-to-buy.php
│   │       │   ├── comparison-table.php
│   │       │   └── product-card.php
│   │       └── assets/
│   │           ├── css/
│   │           │   ├── shgr-design-system.css
│   │           │   ├── review-page.css
│   │           │   ├── homepage.css
│   │           │   └── components.css
│   │           ├── js/
│   │           │   ├── score-animation.js
│   │           │   ├── comparison-table.js
│   │           │   └── affiliate-tracking.js
│   │           └── images/
│   │               ├── logo.svg
│   │               ├── og-image.jpg
│   │               ├── badges/
│   │               └── retailers/
│   └── uploads/
│       └── 2025/
│           └── (product images, NOT Unsplash)
```

---

## 📈 SUCCESS METRICS

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

## 🔧 CUSTOM CSS

```css
/* SHGR Design System */

/* Score Circle */
.score-circle {
  position: relative;
  width: 120px;
  height: 120px;
}

.score-ring {
  transform: rotate(-90deg);
  width: 120px;
  height: 120px;
}

.score-ring circle {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
}

.score-ring .bg {
  stroke: var(--shgr-light);
}

.score-ring .progress {
  stroke: url(#scoreGradient);
  stroke-dasharray: 339.292;
  stroke-dashoffset: calc(339.292 * (1 - var(--score) / 10));
  transition: stroke-dashoffset 1.5s ease-out;
}

/* Rating Colors */
[data-score="10"], [data-score="9"] {
  --score-color: var(--rating-9);
}
[data-score="8"] {
  --score-color: var(--rating-8);
}
[data-score="7"] {
  --score-color: var(--rating-7);
}
[data-score="6"] {
  --score-color: var(--rating-6);
}

/* Badge Styles */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.badge--editors-choice {
  background: var(--badge-editors-choice);
  color: white;
}

.badge--best-value {
  background: var(--badge-best-value);
  color: white;
}

.badge--lab-tested {
  background: var(--shgr-light);
  color: var(--shgr-primary);
  border: 1px solid var(--shgr-primary);
}

/* Product Card */
.product-card {
  background: white;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  transition: all 0.3s ease;
}

.product-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}

.product-card__rating {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 0;
}

.rating-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  font-weight: 700;
  font-size: var(--text-xl);
  color: white;
}

.rating-badge--excellent {
  background: var(--shgr-gradient);
}

/* Score Breakdown Bar */
.breakdown-item {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.item-bar {
  flex: 1;
  height: 8px;
  background: var(--shgr-light);
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: var(--shgr-gradient);
  border-radius: 4px;
  width: calc(var(--score) * 10%);
  transition: width 1s ease-out;
}

/* Category Tiles */
.category-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px 24px;
  background: white;
  border-radius: 16px;
  border: 2px solid var(--shgr-light);
  text-align: center;
  transition: all 0.3s ease;
}

.category-tile:hover {
  border-color: var(--shgr-primary);
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(30, 64, 175, 0.15);
}

.category-tile__icon {
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--shgr-light);
  border-radius: 16px;
  margin-bottom: 16px;
  color: var(--shgr-primary);
}

.category-tile:hover .category-tile__icon {
  background: var(--shgr-gradient);
  color: white;
}

/* Hero Section */
.shgr-hero {
  position: relative;
  min-height: 80vh;
  display: flex;
  align-items: center;
  background: var(--shgr-dark);
  overflow: hidden;
}

.hero-grid {
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(rgba(30,64,175,0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(30,64,175,0.1) 1px, transparent 1px);
  background-size: 60px 60px;
  opacity: 0.5;
}

.hero-glow {
  position: absolute;
  top: 30%;
  left: 50%;
  transform: translateX(-50%);
  width: 50%;
  height: 50%;
  background: radial-gradient(ellipse, var(--shgr-primary)20 0%, transparent 70%);
  filter: blur(80px);
}

/* Trust Indicators */
.shgr-hero__trust {
  display: flex;
  gap: 48px;
  margin-top: 48px;
}

.trust-item {
  text-align: center;
}

.trust-number {
  display: block;
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 700;
  color: var(--shgr-secondary);
}

.trust-label {
  font-size: var(--text-sm);
  color: var(--shgr-muted);
}
```

---

**END OF BLUEPRINT**

*Smart Home Gear Reviews transformation guide. Build the definitive lab-tested smart home review authority.*

**Document Version:** 1.0
**Created:** 2025-12-16
**Author:** Claude (AI Publishing Empire Assistant)

---
