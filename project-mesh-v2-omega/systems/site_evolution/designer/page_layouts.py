"""
Page Layouts — Homepage, category, single post, and about page layout generators.
Uses site-specific CSS variables and brand voice.
"""

import logging
from typing import Dict, List

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)


class PageLayouts:
    """Generate page layout HTML using Evolution CSS framework classes."""

    def generate_homepage_layout(self, site_slug: str) -> List[str]:
        """Generate full homepage as list of HTML sections.

        Returns 8 section HTML strings for ContentDeployer.deploy_homepage().
        """
        config = load_site_config(site_slug)
        brand_name = config.get("name", site_slug)
        brand = config.get("brand", {})
        voice = brand.get("voice", "expert")
        ctas = config.get("ctas", ["Get Started", "Learn More"])
        tagline = brand.get("tagline", f"Your trusted source for expert guidance")

        sections = []

        # 1. Hero
        primary_cta = ctas[0] if ctas else "Get Started"
        secondary_cta = ctas[1] if len(ctas) > 1 else "Learn More"
        sections.append(f"""<section class="evo-hero">
  <h1 class="evo-hero__headline">{brand_name}</h1>
  <p class="evo-hero__subheadline">{tagline}</p>
  <div class="evo-hero__ctas">
    <a href="#featured" class="evo-btn evo-btn-primary">{primary_cta}</a>
    <a href="/about" class="evo-btn evo-btn-secondary" style="color:#fff;border-color:#fff;">{secondary_cta}</a>
  </div>
</section>""")

        # 2. Trust Bar
        sections.append("""<section class="evo-section flex-center gap-lg" style="padding:var(--space-xl) var(--space-xl);">
  <div class="flex-center gap-lg flex-wrap" style="justify-content:center;width:100%;">
    <div class="text-center" style="min-width:120px;">
      <div style="font-size:var(--font-size-3xl);font-weight:700;color:var(--color-primary);">100+</div>
      <div class="text-muted" style="font-size:var(--font-size-sm);">Expert Articles</div>
    </div>
    <div class="text-center" style="min-width:120px;">
      <div style="font-size:var(--font-size-3xl);font-weight:700;color:var(--color-primary);">50K+</div>
      <div class="text-muted" style="font-size:var(--font-size-sm);">Monthly Readers</div>
    </div>
    <div class="text-center" style="min-width:120px;">
      <div style="font-size:var(--font-size-3xl);font-weight:700;color:var(--color-primary);">4.8/5</div>
      <div class="text-muted" style="font-size:var(--font-size-sm);">Reader Rating</div>
    </div>
  </div>
</section>""")

        # 3. Featured Content
        sections.append(f"""<section id="featured" class="evo-section">
  <h2 class="evo-section__title">Featured Articles</h2>
  <div class="evo-grid evo-grid--3">
    <article class="evo-card" data-animate>
      <div class="evo-card__body">
        <span class="badge">Featured</span>
        <h3 class="evo-card__title">Latest from {brand_name}</h3>
        <p class="evo-card__excerpt">Discover our most popular and impactful content, curated for you.</p>
        <div class="evo-card__meta"><span>5 min read</span></div>
      </div>
    </article>
    <article class="evo-card" data-animate>
      <div class="evo-card__body">
        <span class="badge">Popular</span>
        <h3 class="evo-card__title">Trending Now</h3>
        <p class="evo-card__excerpt">See what our community is reading and sharing this week.</p>
        <div class="evo-card__meta"><span>4 min read</span></div>
      </div>
    </article>
    <article class="evo-card" data-animate>
      <div class="evo-card__body">
        <span class="badge">New</span>
        <h3 class="evo-card__title">Just Published</h3>
        <p class="evo-card__excerpt">Fresh content from our expert team, updated regularly.</p>
        <div class="evo-card__meta"><span>6 min read</span></div>
      </div>
    </article>
  </div>
</section>""")

        # 4. Category Showcase
        sections.append("""<section class="evo-section evo-section--alt">
  <h2 class="evo-section__title">Explore Topics</h2>
  <div class="evo-grid evo-grid--4">
    <a href="/category/getting-started" class="evo-card text-center p-xl" style="text-decoration:none;">
      <div style="font-size:2.5rem;margin-bottom:var(--space-md);">&#128218;</div>
      <h3 style="font-size:var(--font-size-lg);">Getting Started</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Begin your journey here</p>
    </a>
    <a href="/category/guides" class="evo-card text-center p-xl" style="text-decoration:none;">
      <div style="font-size:2.5rem;margin-bottom:var(--space-md);">&#128161;</div>
      <h3 style="font-size:var(--font-size-lg);">In-Depth Guides</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Deep dives into topics</p>
    </a>
    <a href="/category/reviews" class="evo-card text-center p-xl" style="text-decoration:none;">
      <div style="font-size:2.5rem;margin-bottom:var(--space-md);">&#11088;</div>
      <h3 style="font-size:var(--font-size-lg);">Reviews</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Honest expert reviews</p>
    </a>
    <a href="/category/tips" class="evo-card text-center p-xl" style="text-decoration:none;">
      <div style="font-size:2.5rem;margin-bottom:var(--space-md);">&#127919;</div>
      <h3 style="font-size:var(--font-size-lg);">Quick Tips</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Actionable advice</p>
    </a>
  </div>
</section>""")

        # 5. Latest Articles (dynamically fetched from WP REST API)
        domain = config.get("domain", "")
        sections.append(f"""<section class="evo-section">
  <h2 class="evo-section__title">Latest Articles</h2>
  <div class="evo-grid evo-grid--3" id="latest-posts">
    <article class="evo-card" data-animate>
      <div class="evo-card__body">
        <span class="badge">Latest</span>
        <h3 class="evo-card__title">Loading...</h3>
        <p class="evo-card__excerpt">Fetching the latest articles for you.</p>
      </div>
    </article>
  </div>
  <script>
  (function() {{
    var container = document.getElementById('latest-posts');
    if (!container) return;
    fetch('https://{domain}/wp-json/wp/v2/posts?per_page=6&_fields=id,title,excerpt,link,date,featured_media,_links&_embed=wp:featuredmedia')
      .then(function(r) {{ return r.json(); }})
      .then(function(posts) {{
        if (!posts || !posts.length) return;
        container.innerHTML = '';
        posts.forEach(function(post) {{
          var title = post.title ? post.title.rendered : '';
          var excerpt = post.excerpt ? post.excerpt.rendered.replace(/<[^>]+>/g, '').slice(0, 120) + '...' : '';
          var link = post.link || '#';
          var img = '';
          try {{ img = post._embedded['wp:featuredmedia'][0].source_url; }} catch(e) {{}}
          var card = document.createElement('article');
          card.className = 'evo-card';
          card.setAttribute('data-animate', '');
          card.innerHTML = (img ? '<img class="evo-card__image" src="' + img + '" alt="' + title + '" loading="lazy">' : '') +
            '<div class="evo-card__body">' +
            '<h3 class="evo-card__title"><a href="' + link + '">' + title + '</a></h3>' +
            '<p class="evo-card__excerpt">' + excerpt + '</p>' +
            '</div>';
          container.appendChild(card);
        }});
      }})
      .catch(function() {{}});
  }})();
  </script>
</section>""")

        # 6. Newsletter CTA
        sections.append(f"""<section class="evo-section">
  <div class="evo-newsletter">
    <h2 style="color:#fff;">Stay in the Loop</h2>
    <p style="opacity:0.9;">Get the latest from {brand_name} delivered straight to your inbox. No spam, ever.</p>
    <form class="evo-newsletter__form" action="#" method="post">
      <input type="email" class="evo-newsletter__input" placeholder="Your email address" required>
      <button type="submit" class="evo-btn evo-btn-primary" style="background:#fff;color:var(--color-primary);">Subscribe</button>
    </form>
    <p style="font-size:var(--font-size-xs);opacity:0.7;margin-top:var(--space-md);">We respect your privacy. Unsubscribe anytime.</p>
  </div>
</section>""")

        # 7. About Preview
        sections.append(f"""<section class="evo-section evo-section--alt text-center">
  <h2>About {brand_name}</h2>
  <p class="max-w-content mx-auto" style="font-size:var(--font-size-lg);color:var(--color-text-muted);">
    We're passionate about delivering expert, well-researched content that
    helps you make informed decisions. Every article is crafted with care.
  </p>
  <a href="/about" class="evo-btn evo-btn-primary" style="margin-top:var(--space-xl);">Learn More About Us</a>
</section>""")

        # 8. Footer
        year = __import__("datetime").date.today().year
        sections.append(f"""<footer class="evo-footer">
  <div class="evo-footer__grid">
    <div>
      <div class="evo-footer__brand">{brand_name}</div>
      <p class="text-muted" style="font-size:var(--font-size-sm);">{tagline}</p>
    </div>
    <div>
      <h4 style="font-size:var(--font-size-sm);margin-bottom:var(--space-md);">Categories</h4>
      <ul class="evo-footer__links">
        <li><a href="/category/getting-started">Getting Started</a></li>
        <li><a href="/category/guides">Guides</a></li>
        <li><a href="/category/reviews">Reviews</a></li>
        <li><a href="/category/tips">Tips</a></li>
      </ul>
    </div>
    <div>
      <h4 style="font-size:var(--font-size-sm);margin-bottom:var(--space-md);">Legal</h4>
      <ul class="evo-footer__links">
        <li><a href="/about">About</a></li>
        <li><a href="/contact">Contact</a></li>
        <li><a href="/privacy-policy">Privacy Policy</a></li>
        <li><a href="/affiliate-disclosure">Affiliate Disclosure</a></li>
      </ul>
    </div>
  </div>
  <div class="evo-footer__copyright">&copy; {year} {brand_name}. All rights reserved.</div>
</footer>""")

        return sections

    def generate_about_page(self, site_slug: str) -> str:
        """Generate a branded about page."""
        config = load_site_config(site_slug)
        brand_name = config.get("name", site_slug)
        voice = config.get("brand", {}).get("voice", "expert guide")

        return f"""<div class="evo-section max-w-content mx-auto">
  <h1>About {brand_name}</h1>

  <h2>Our Mission</h2>
  <p>At {brand_name}, our mission is simple: to be the most trusted, comprehensive resource
  in our field. We combine deep expertise with accessible writing to help you navigate
  complex topics with confidence.</p>

  <h2>Why Trust Us</h2>
  <div class="evo-grid evo-grid--3" style="margin:var(--space-xl) 0;">
    <div class="evo-card p-lg text-center">
      <div style="font-size:2rem;">&#128214;</div>
      <h3 style="font-size:var(--font-size-lg);">Thoroughly Researched</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Every article backed by real data and expert knowledge.</p>
    </div>
    <div class="evo-card p-lg text-center">
      <div style="font-size:2rem;">&#9989;</div>
      <h3 style="font-size:var(--font-size-lg);">Regularly Updated</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Content reviewed and refreshed to stay current.</p>
    </div>
    <div class="evo-card p-lg text-center">
      <div style="font-size:2rem;">&#128588;</div>
      <h3 style="font-size:var(--font-size-lg);">Reader-First</h3>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Honest opinions, transparent affiliate disclosures.</p>
    </div>
  </div>

  <h2>Editorial Standards</h2>
  <ul>
    <li>All content is original and never AI-generated without human review</li>
    <li>Affiliate relationships never influence our editorial judgment</li>
    <li>We correct errors promptly when discovered</li>
    <li>We clearly label sponsored and affiliate content</li>
  </ul>

  <div class="evo-cta evo-cta--subtle" style="margin-top:var(--space-2xl);">
    <h3>Have Questions?</h3>
    <p>We'd love to hear from you.</p>
    <a href="/contact" class="evo-btn evo-btn-primary">Contact Us</a>
  </div>
</div>"""
