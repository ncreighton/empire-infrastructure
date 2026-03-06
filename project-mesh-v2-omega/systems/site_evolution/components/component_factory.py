"""
Component Factory — Generate 12 component types with HTML + CSS + JS per site.
Each component reads brand config for colors, fonts, voice, CTAs.
"""

import logging
from typing import Dict, Optional

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)

COMPONENT_TYPES = [
    "hero_section", "navigation_header", "article_cards", "newsletter_signup",
    "footer_section", "sidebar_widgets", "author_box", "breadcrumbs",
    "table_of_contents", "related_posts", "comparison_table", "cta_sections",
    # v2.0 additions
    "accordion_tabs", "modal_popup", "image_gallery", "search_bar",
    "sticky_header", "toast_notification", "loading_skeleton", "pagination",
    "testimonials", "timeline", "stat_counter", "social_share_bar",
    "reading_progress_bar", "back_to_top", "cookie_consent", "dark_mode_toggle",
]


class ComponentFactory:
    """Generate branded components for any site."""

    def generate_component(self, site_slug: str, component_type: str,
                           variant: str = "default") -> Dict:
        """Generate a single component. Returns {html, css, js, snippet_name}."""
        config = load_site_config(site_slug)
        brand_name = config.get("name", site_slug)
        ctas = config.get("ctas", ["Get Started"])

        method_name = f"_gen_{component_type}"
        if hasattr(self, method_name):
            result = getattr(self, method_name)(site_slug, config, variant)
        else:
            result = {"html": f"<!-- {component_type} not implemented -->", "css": "", "js": ""}

        result.setdefault("snippet_name", f"{site_slug[:4]}-{component_type}-v1")
        return result

    def generate_all_components(self, site_slug: str) -> Dict[str, Dict]:
        """Generate all 12 component types for a site."""
        results = {}
        for ct in COMPONENT_TYPES:
            try:
                results[ct] = self.generate_component(site_slug, ct)
            except Exception as e:
                log.error("Failed to generate %s for %s: %s", ct, site_slug, e)
                results[ct] = {"error": str(e)}
        return results

    def _gen_hero_section(self, site_slug: str, config: Dict,
                          variant: str) -> Dict:
        brand = config.get("name", site_slug)
        ctas = config.get("ctas", ["Get Started", "Learn More"])
        tagline = config.get("brand", {}).get("tagline", "Your trusted guide")

        html = f"""<section class="evo-hero">
  <div class="evo-hero__inner">
    <span class="evo-hero__badge">Welcome to {brand}</span>
    <h1 class="evo-hero__headline">{brand}</h1>
    <p class="evo-hero__subheadline">{tagline}</p>
    <div class="evo-hero__ctas">
      <a href="#featured" class="evo-btn evo-btn-primary">{ctas[0]}</a>
      {"<a href='/about' class='evo-btn evo-btn-secondary' style='color:#fff;border-color:#fff;'>" + ctas[1] + "</a>" if len(ctas) > 1 else ""}
    </div>
    <div class="evo-hero__trust" style="margin-top:var(--space-xl);opacity:0.8;font-size:var(--font-size-sm);">
      Trusted by thousands of readers
    </div>
  </div>
</section>"""
        css = """.evo-hero__inner { max-width: 800px; margin: 0 auto; }
.evo-hero__badge { display: inline-block; padding: 4px 16px; border-radius: var(--radius-full, 9999px);
  background: rgba(255,255,255,0.15); font-size: var(--font-size-sm); margin-bottom: var(--space-lg);
  backdrop-filter: blur(4px); letter-spacing: 0.05em; text-transform: uppercase; }"""
        return {"html": html, "css": css, "js": ""}

    def _gen_navigation_header(self, site_slug: str, config: Dict,
                               variant: str) -> Dict:
        brand = config.get("name", site_slug)
        ctas = config.get("ctas", ["Subscribe"])

        html = f"""<nav class="evo-nav" role="navigation">
  <a href="/" class="evo-nav__logo">{brand}</a>
  <ul class="evo-nav__links">
    <li><a href="/">Home</a></li>
    <li><a href="/blog">Blog</a></li>
    <li><a href="/about">About</a></li>
    <li><a href="/contact">Contact</a></li>
  </ul>
  <a href="#newsletter" class="evo-btn evo-btn-primary" style="padding:8px 20px;font-size:var(--font-size-sm);">{ctas[0] if ctas else 'Subscribe'}</a>
  <button class="evo-nav__hamburger" aria-label="Menu" style="display:none;background:none;border:none;font-size:1.5rem;cursor:pointer;">&#9776;</button>
</nav>"""

        js = """(function(){
  var hamburger = document.querySelector('.evo-nav__hamburger');
  var links = document.querySelector('.evo-nav__links');
  if(hamburger && links) {
    hamburger.addEventListener('click', function() {
      links.style.display = links.style.display === 'flex' ? 'none' : 'flex';
    });
  }
  if(window.innerWidth <= 768 && hamburger) { hamburger.style.display = 'block'; }
  window.addEventListener('resize', function() {
    if(hamburger) hamburger.style.display = window.innerWidth <= 768 ? 'block' : 'none';
    if(links && window.innerWidth > 768) links.style.display = 'flex';
  });
})();"""
        return {"html": html, "css": "", "js": js}

    def _gen_article_cards(self, site_slug: str, config: Dict,
                           variant: str) -> Dict:
        html = """<template id="evo-card-template">
  <article class="evo-card" data-animate>
    <img class="evo-card__image" src="" alt="" loading="lazy">
    <div class="evo-card__body">
      <span class="badge"></span>
      <h3 class="evo-card__title"><a href=""></a></h3>
      <p class="evo-card__excerpt"></p>
      <div class="evo-card__meta">
        <span class="evo-card__date"></span>
        <span>&middot;</span>
        <span class="evo-card__read-time"></span>
      </div>
    </div>
  </article>
</template>"""
        return {"html": html, "css": "", "js": ""}

    def _gen_newsletter_signup(self, site_slug: str, config: Dict,
                               variant: str) -> Dict:
        brand = config.get("name", site_slug)
        html = f"""<div class="evo-newsletter" id="newsletter">
  <h2 style="color:#fff;">Join the {brand} Community</h2>
  <p style="opacity:0.9;">Expert insights delivered to your inbox. Free forever.</p>
  <form class="evo-newsletter__form" action="#" method="post">
    <input type="email" class="evo-newsletter__input" name="email" placeholder="your@email.com" required>
    <button type="submit" class="evo-btn evo-btn-primary" style="background:#fff;color:var(--color-primary);">Subscribe</button>
  </form>
  <p style="font-size:var(--font-size-xs);opacity:0.7;margin-top:var(--space-sm);">No spam. Unsubscribe anytime.</p>
</div>"""
        return {"html": html, "css": "", "js": ""}

    def _gen_footer_section(self, site_slug: str, config: Dict,
                            variant: str) -> Dict:
        brand = config.get("name", site_slug)
        year = __import__("datetime").date.today().year
        html = f"""<footer class="evo-footer">
  <div class="evo-footer__grid">
    <div>
      <div class="evo-footer__brand">{brand}</div>
      <p class="text-muted" style="font-size:var(--font-size-sm);">Your trusted resource for expert guidance.</p>
    </div>
    <div>
      <h4 style="font-size:var(--font-size-sm);margin-bottom:var(--space-md);">Navigate</h4>
      <ul class="evo-footer__links">
        <li><a href="/">Home</a></li>
        <li><a href="/blog">Blog</a></li>
        <li><a href="/about">About</a></li>
        <li><a href="/contact">Contact</a></li>
      </ul>
    </div>
    <div>
      <h4 style="font-size:var(--font-size-sm);margin-bottom:var(--space-md);">Legal</h4>
      <ul class="evo-footer__links">
        <li><a href="/privacy-policy">Privacy Policy</a></li>
        <li><a href="/terms-of-service">Terms</a></li>
        <li><a href="/affiliate-disclosure">Affiliate Disclosure</a></li>
        <li><a href="/cookie-policy">Cookies</a></li>
      </ul>
    </div>
  </div>
  <div class="evo-footer__copyright">&copy; {year} {brand}. All rights reserved.</div>
</footer>"""
        return {"html": html, "css": "", "js": ""}

    def _gen_sidebar_widgets(self, site_slug: str, config: Dict,
                             variant: str) -> Dict:
        brand = config.get("name", site_slug)
        html = f"""<aside class="evo-sidebar">
  <div class="evo-card p-lg" style="margin-bottom:var(--space-xl);">
    <h3 style="font-size:var(--font-size-lg);">About {brand}</h3>
    <p class="text-muted" style="font-size:var(--font-size-sm);">Expert guidance you can trust.</p>
    <a href="/about" class="evo-btn evo-btn-secondary" style="margin-top:var(--space-md);padding:8px 16px;font-size:var(--font-size-sm);">Learn More</a>
  </div>
  <div class="evo-newsletter" style="padding:var(--space-lg);border-radius:var(--radius-md);">
    <h3 style="color:#fff;font-size:var(--font-size-lg);">Newsletter</h3>
    <form class="evo-newsletter__form" style="flex-direction:column;" action="#" method="post">
      <input type="email" class="evo-newsletter__input" placeholder="Email" required>
      <button type="submit" class="evo-btn" style="background:#fff;color:var(--color-primary);width:100%;">Subscribe</button>
    </form>
  </div>
</aside>"""
        return {"html": html, "css": "", "js": ""}

    def _gen_author_box(self, site_slug: str, config: Dict,
                        variant: str) -> Dict:
        brand = config.get("name", site_slug)
        html = f"""<div class="evo-author">
  <img class="evo-author__avatar" src="/wp-content/uploads/author-avatar.jpg" alt="{brand} Editorial Team" loading="lazy">
  <div class="evo-author__info">
    <div class="evo-author__name">{brand} Editorial Team</div>
    <p class="evo-author__bio">Our team of subject matter experts brings years of hands-on experience to every article we publish. All content is thoroughly researched, fact-checked, and regularly updated.</p>
    <div class="evo-author__meta" style="display:flex;gap:var(--space-md);margin-top:var(--space-sm);font-size:var(--font-size-xs);color:var(--color-text-muted);">
      <span>Verified Expert</span>
      <span>&middot;</span>
      <a href="/about" style="color:var(--color-primary);">About Us</a>
    </div>
  </div>
</div>"""
        css = """.evo-author { display: flex; gap: var(--space-lg); padding: var(--space-xl);
  background: var(--color-bg-alt); border-radius: var(--radius-md); margin: var(--space-2xl) 0;
  border: 1px solid var(--color-divider, #e5e7eb); }
.evo-author__avatar { width: 80px; height: 80px; border-radius: 50%; object-fit: cover; flex-shrink: 0; }
.evo-author__name { font-weight: 700; font-size: var(--font-size-lg); margin-bottom: var(--space-xs); }
.evo-author__bio { color: var(--color-text-muted); font-size: var(--font-size-sm); line-height: 1.6; margin: 0; }
@media (max-width: 640px) {
  .evo-author { flex-direction: column; align-items: center; text-align: center; }
}"""
        return {"html": html, "css": css, "js": ""}

    def _gen_breadcrumbs(self, site_slug: str, config: Dict,
                         variant: str) -> Dict:
        html = """<nav class="evo-breadcrumbs" aria-label="Breadcrumb">
  <a href="/">Home</a>
  <span class="evo-breadcrumbs__sep">/</span>
  <a href="/category">Category</a>
  <span class="evo-breadcrumbs__sep">/</span>
  <span aria-current="page">Current Page</span>
</nav>"""
        return {"html": html, "css": "", "js": ""}

    def _gen_table_of_contents(self, site_slug: str, config: Dict,
                               variant: str) -> Dict:
        js = """(function(){
  var content = document.querySelector('.entry-content, .post-content, article');
  if(!content) return;
  var headings = content.querySelectorAll('h2, h3');
  if(headings.length < 3) return;

  var toc = document.createElement('div');
  toc.className = 'evo-toc';
  toc.innerHTML = '<div class="evo-toc__title">Table of Contents</div>';
  var ol = document.createElement('ol');

  headings.forEach(function(h, i) {
    if(!h.id) h.id = 'section-' + i;
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    if(h.tagName === 'H3') li.style.paddingLeft = '1em';
    li.appendChild(a);
    ol.appendChild(li);
  });

  toc.appendChild(ol);
  var firstH2 = content.querySelector('h2');
  if(firstH2) firstH2.parentNode.insertBefore(toc, firstH2);
})();"""
        return {"html": "", "css": "", "js": js, "location": "site_wide_footer"}

    def _gen_related_posts(self, site_slug: str, config: Dict,
                           variant: str) -> Dict:
        domain = config.get("domain", "")
        html = """<section class="evo-related" style="padding-top:var(--space-2xl);border-top:1px solid var(--color-divider,#e5e7eb);">
  <h2 class="evo-section__title" style="font-size:var(--font-size-2xl);">You Might Also Like</h2>
  <div class="evo-grid evo-grid--3" id="evo-related-posts"></div>
</section>"""
        js = f"""(function(){{
  var container = document.getElementById('evo-related-posts');
  if(!container) return;
  var cats = document.querySelector('meta[property="article:section"]');
  var catSlug = cats ? cats.getAttribute('content').toLowerCase().replace(/\\s+/g,'-') : '';
  var postId = document.querySelector('article[id^="post-"]');
  var excludeId = postId ? postId.id.replace('post-','') : '';
  var url = 'https://{domain}/wp-json/wp/v2/posts?per_page=3&_fields=id,title,excerpt,link,featured_media,_links&_embed=wp:featuredmedia';
  if(excludeId) url += '&exclude=' + excludeId;
  fetch(url)
    .then(function(r){{ return r.json(); }})
    .then(function(posts){{
      if(!posts||!posts.length) return;
      container.innerHTML='';
      posts.forEach(function(p){{
        var t=p.title?p.title.rendered:'';
        var e=p.excerpt?p.excerpt.rendered.replace(/<[^>]+>/g,'').slice(0,100)+'...':'';
        var l=p.link||'#';
        var img='';
        try{{img=p._embedded['wp:featuredmedia'][0].source_url;}}catch(ex){{}}
        var card=document.createElement('article');
        card.className='evo-card';
        card.innerHTML=(img?'<img class="evo-card__image" src="'+img+'" alt="'+t+'" loading="lazy">':'')+
          '<div class="evo-card__body">'+
          '<h3 class="evo-card__title"><a href="'+l+'">'+t+'</a></h3>'+
          '<p class="evo-card__excerpt">'+e+'</p></div>';
        container.appendChild(card);
      }});
    }}).catch(function(){{}});
}})();"""
        return {"html": html, "css": "", "js": js, "location": "site_wide_footer"}

    def _gen_comparison_table(self, site_slug: str, config: Dict,
                              variant: str) -> Dict:
        brand = config.get("name", site_slug)
        amazon_tag = config.get("amazon_tag", "")
        html = f"""<div class="evo-comparison-wrapper" style="overflow-x:auto;margin:var(--space-xl) 0;">
  <div style="font-size:var(--font-size-sm);color:var(--color-text-muted);margin-bottom:var(--space-sm);">
    Comparison reviewed by {brand} editorial team
  </div>
  <table class="evo-comparison">
    <thead>
      <tr>
        <th style="min-width:180px;">Product</th>
        <th style="min-width:100px;">Rating</th>
        <th style="min-width:80px;">Price</th>
        <th style="min-width:150px;">Best For</th>
        <th style="min-width:120px;">Action</th>
      </tr>
    </thead>
    <tbody id="evo-comparison-body">
      <tr>
        <td><strong>Example Product</strong><br><span class="text-muted" style="font-size:var(--font-size-xs);">Brand Name</span></td>
        <td><span style="color:var(--color-primary);font-weight:700;">4.8</span>/5</td>
        <td>$49.99</td>
        <td>Best Overall</td>
        <td><a href="#" class="evo-btn evo-btn-primary" style="padding:6px 16px;font-size:var(--font-size-sm);">Check Price</a></td>
      </tr>
    </tbody>
  </table>
  <p style="font-size:var(--font-size-xs);color:var(--color-text-muted);margin-top:var(--space-sm);">
    Prices and availability may vary. As an Amazon Associate, we earn from qualifying purchases.
  </p>
</div>"""
        css = """.evo-comparison { width: 100%; border-collapse: collapse; }
.evo-comparison th { background: var(--color-bg-alt); padding: var(--space-md); text-align: left;
  font-size: var(--font-size-sm); font-weight: 600; border-bottom: 2px solid var(--color-primary); }
.evo-comparison td { padding: var(--space-md); border-bottom: 1px solid var(--color-divider, #e5e7eb);
  vertical-align: middle; }
.evo-comparison tr:hover { background: var(--color-bg-alt); }
.evo-comparison tr:last-child td { border-bottom: none; }
@media (max-width: 768px) {
  .evo-comparison { font-size: var(--font-size-sm); }
  .evo-comparison th, .evo-comparison td { padding: var(--space-sm); }
}"""
        return {"html": html, "css": css, "js": ""}

    def _gen_cta_sections(self, site_slug: str, config: Dict,
                          variant: str) -> Dict:
        brand = config.get("name", site_slug)
        ctas = config.get("ctas", ["Get Started"])
        html = f"""<div class="evo-cta evo-cta--gradient">
  <div class="evo-cta__inner">
    <h3 style="color:#fff;font-size:var(--font-size-2xl);">Ready to take the next step?</h3>
    <p style="color:#fff;opacity:0.9;max-width:500px;margin:var(--space-md) auto;">Join thousands of readers who trust {brand} for expert guidance.</p>
    <a href="#newsletter" class="evo-btn" style="background:#fff;color:var(--color-primary);margin-top:var(--space-lg);padding:12px 32px;font-weight:600;">{ctas[0]}</a>
  </div>
</div>"""
        css = """.evo-cta--gradient { background: linear-gradient(135deg, var(--color-primary), var(--color-secondary, var(--color-primary)));
  padding: var(--space-3xl) var(--space-xl); border-radius: var(--radius-lg); text-align: center;
  margin: var(--space-2xl) 0; }
.evo-cta__inner { max-width: 600px; margin: 0 auto; }
.evo-cta--subtle { background: var(--color-bg-alt); padding: var(--space-2xl); border-radius: var(--radius-md);
  text-align: center; border: 1px solid var(--color-divider, #e5e7eb); }"""
        return {"html": html, "css": css, "js": ""}

    # -- v2.0 Component Types --

    def _gen_accordion_tabs(self, site_slug: str, config: Dict,
                            variant: str) -> Dict:
        html = """<div class="evo-accordion">
  <details class="evo-accordion__item" open>
    <summary class="evo-accordion__trigger">Section One</summary>
    <div class="evo-accordion__content"><p>Content for section one.</p></div>
  </details>
  <details class="evo-accordion__item">
    <summary class="evo-accordion__trigger">Section Two</summary>
    <div class="evo-accordion__content"><p>Content for section two.</p></div>
  </details>
  <details class="evo-accordion__item">
    <summary class="evo-accordion__trigger">Section Three</summary>
    <div class="evo-accordion__content"><p>Content for section three.</p></div>
  </details>
</div>"""
        css = """.evo-accordion { margin: var(--space-xl) 0; }
.evo-accordion__item { border: 1px solid var(--color-divider, #e5e7eb); border-radius: var(--radius-md); margin-bottom: var(--space-sm); overflow: hidden; }
.evo-accordion__trigger { padding: var(--space-md) var(--space-lg); font-weight: 600; cursor: pointer; list-style: none; display: flex; justify-content: space-between; align-items: center; }
.evo-accordion__trigger::-webkit-details-marker { display: none; }
.evo-accordion__trigger::after { content: '+'; font-size: 1.2em; transition: transform 0.2s; }
details[open] .evo-accordion__trigger::after { content: '−'; }
.evo-accordion__content { padding: 0 var(--space-lg) var(--space-lg); }"""
        return {"html": html, "css": css, "js": ""}

    def _gen_modal_popup(self, site_slug: str, config: Dict,
                         variant: str) -> Dict:
        html = """<div id="evo-modal" class="evo-modal" role="dialog" aria-modal="true" style="display:none;">
  <div class="evo-modal__backdrop"></div>
  <div class="evo-modal__content">
    <button class="evo-modal__close" aria-label="Close">&times;</button>
    <div class="evo-modal__body">
      <h3>Modal Title</h3>
      <p>Modal content goes here.</p>
    </div>
  </div>
</div>"""
        css = """.evo-modal { position: fixed; inset: 0; z-index: 99999; display: flex; align-items: center; justify-content: center; }
.evo-modal__backdrop { position: absolute; inset: 0; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); }
.evo-modal__content { position: relative; background: var(--color-bg, #fff); border-radius: var(--radius-lg); padding: var(--space-2xl); max-width: 560px; width: 90%; max-height: 90vh; overflow-y: auto; animation: evo-popIn 0.3s ease; }
.evo-modal__close { position: absolute; top: 12px; right: 16px; background: none; border: none; font-size: 28px; cursor: pointer; color: var(--color-text-muted); }
@keyframes evo-popIn { from { opacity:0; transform:scale(0.9); } to { opacity:1; transform:scale(1); } }"""
        js = """(function(){
  document.querySelectorAll('[data-evo-modal]').forEach(function(trigger){
    trigger.addEventListener('click', function(e){
      e.preventDefault();
      var modal = document.getElementById(this.getAttribute('data-evo-modal'));
      if(modal) modal.style.display='flex';
    });
  });
  document.querySelectorAll('.evo-modal__close,.evo-modal__backdrop').forEach(function(el){
    el.addEventListener('click', function(){
      this.closest('.evo-modal').style.display='none';
    });
  });
  document.addEventListener('keydown', function(e){
    if(e.key==='Escape') document.querySelectorAll('.evo-modal').forEach(function(m){ m.style.display='none'; });
  });
})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_image_gallery(self, site_slug: str, config: Dict,
                           variant: str) -> Dict:
        html = """<div class="evo-gallery">
  <div class="evo-gallery__grid" id="evo-gallery-grid">
    <!-- Gallery items populated dynamically or via shortcode -->
  </div>
  <div class="evo-gallery__lightbox" id="evo-lightbox" style="display:none;">
    <div class="evo-gallery__lightbox-backdrop"></div>
    <img class="evo-gallery__lightbox-img" src="" alt="">
    <button class="evo-gallery__lightbox-close" aria-label="Close">&times;</button>
  </div>
</div>"""
        css = """.evo-gallery__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--space-md); }
.evo-gallery__grid img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: var(--radius-md); cursor: pointer; transition: transform 0.2s; }
.evo-gallery__grid img:hover { transform: scale(1.03); }
.evo-gallery__lightbox { position: fixed; inset: 0; z-index: 99999; display: flex; align-items: center; justify-content: center; }
.evo-gallery__lightbox-backdrop { position: absolute; inset: 0; background: rgba(0,0,0,0.9); }
.evo-gallery__lightbox-img { position: relative; max-width: 90vw; max-height: 90vh; border-radius: var(--radius-md); }
.evo-gallery__lightbox-close { position: absolute; top: 20px; right: 20px; background: none; border: none; color: #fff; font-size: 36px; cursor: pointer; }"""
        js = """(function(){
  var lb = document.getElementById('evo-lightbox');
  if(!lb) return;
  document.querySelectorAll('.evo-gallery__grid img').forEach(function(img){
    img.addEventListener('click', function(){
      lb.querySelector('img').src = this.src;
      lb.querySelector('img').alt = this.alt;
      lb.style.display = 'flex';
    });
  });
  lb.querySelector('.evo-gallery__lightbox-close').addEventListener('click', function(){ lb.style.display='none'; });
  lb.querySelector('.evo-gallery__lightbox-backdrop').addEventListener('click', function(){ lb.style.display='none'; });
})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_search_bar(self, site_slug: str, config: Dict,
                        variant: str) -> Dict:
        domain = config.get("domain", "")
        html = """<div class="evo-search">
  <input type="search" class="evo-search__input" id="evo-search-input" placeholder="Search articles..." aria-label="Search">
  <div class="evo-search__results" id="evo-search-results" style="display:none;"></div>
</div>"""
        css = """.evo-search { position: relative; max-width: 400px; }
.evo-search__input { width: 100%; padding: 12px 16px 12px 40px; border: 2px solid var(--color-divider, #e5e7eb); border-radius: var(--radius-md); font-size: var(--font-size-base); background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%23999' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85zm-5.242.656a5 5 0 1 1 0-10 5 5 0 0 1 0 10z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: 12px center; }
.evo-search__input:focus { border-color: var(--color-primary); outline: none; }
.evo-search__results { position: absolute; top: 100%; left: 0; right: 0; background: var(--color-bg, #fff); border: 1px solid var(--color-divider); border-radius: var(--radius-md); margin-top: 4px; max-height: 300px; overflow-y: auto; box-shadow: var(--shadow-lg); z-index: 100; }
.evo-search__results a { display: block; padding: 10px 16px; color: var(--color-text); border-bottom: 1px solid var(--color-divider, #f0f0f0); }
.evo-search__results a:hover { background: var(--color-bg-alt); }"""
        js = f"""(function(){{
  var input = document.getElementById('evo-search-input');
  var results = document.getElementById('evo-search-results');
  if(!input||!results) return;
  var timer;
  input.addEventListener('input', function(){{
    clearTimeout(timer);
    var q = this.value.trim();
    if(q.length<3){{ results.style.display='none'; return; }}
    timer = setTimeout(function(){{
      fetch('https://{domain}/wp-json/wp/v2/posts?search='+encodeURIComponent(q)+'&per_page=5&_fields=title,link')
        .then(function(r){{ return r.json(); }})
        .then(function(posts){{
          if(!posts.length){{ results.innerHTML='<div style="padding:12px;color:#999;">No results</div>'; }}
          else {{ results.innerHTML=posts.map(function(p){{ return '<a href="'+p.link+'">'+p.title.rendered+'</a>'; }}).join(''); }}
          results.style.display='block';
        }}).catch(function(){{ results.style.display='none'; }});
    }}, 300);
  }});
  document.addEventListener('click', function(e){{ if(!e.target.closest('.evo-search')) results.style.display='none'; }});
}})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_sticky_header(self, site_slug: str, config: Dict,
                           variant: str) -> Dict:
        js = """(function(){
  var nav = document.querySelector('.evo-nav');
  if(!nav) return;
  var offset = nav.offsetTop + nav.offsetHeight;
  window.addEventListener('scroll', function(){
    if(window.scrollY > offset) { nav.classList.add('evo-nav--sticky'); }
    else { nav.classList.remove('evo-nav--sticky'); }
  });
})();"""
        css = """.evo-nav--sticky { position: fixed; top: 0; left: 0; right: 0; box-shadow: var(--shadow-md); background: var(--color-bg); animation: evo-slideDown 0.3s ease; }
@keyframes evo-slideDown { from { transform: translateY(-100%); } to { transform: translateY(0); } }
body.admin-bar .evo-nav--sticky { top: 32px; }"""
        return {"html": "", "css": css, "js": js}

    def _gen_toast_notification(self, site_slug: str, config: Dict,
                                variant: str) -> Dict:
        html = """<div id="evo-toast-container" class="evo-toast-container"></div>"""
        css = """.evo-toast-container { position: fixed; bottom: 20px; right: 20px; z-index: 99998; display: flex; flex-direction: column; gap: 8px; }
.evo-toast { padding: 14px 20px; border-radius: var(--radius-md); color: #fff; font-size: var(--font-size-sm); min-width: 280px; animation: evo-slideUp 0.3s ease; display: flex; justify-content: space-between; align-items: center; }
.evo-toast--success { background: #059669; }
.evo-toast--error { background: #DC2626; }
.evo-toast--info { background: var(--color-primary, #6B46C1); }
.evo-toast__close { background: none; border: none; color: #fff; cursor: pointer; font-size: 18px; margin-left: 12px; }
@keyframes evo-slideUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }"""
        js = """window.evoToast = function(msg, type) {
  type = type || 'info';
  var container = document.getElementById('evo-toast-container');
  if(!container) return;
  var toast = document.createElement('div');
  toast.className = 'evo-toast evo-toast--' + type;
  toast.innerHTML = '<span>' + msg + '</span><button class="evo-toast__close">&times;</button>';
  container.appendChild(toast);
  toast.querySelector('.evo-toast__close').addEventListener('click', function(){ toast.remove(); });
  setTimeout(function(){ toast.remove(); }, 5000);
};"""
        return {"html": html, "css": css, "js": js}

    def _gen_loading_skeleton(self, site_slug: str, config: Dict,
                              variant: str) -> Dict:
        css = """.evo-skeleton { background: linear-gradient(90deg, var(--color-bg-alt, #f0f0f0) 25%, #e0e0e0 50%, var(--color-bg-alt, #f0f0f0) 75%); background-size: 200% 100%; animation: evo-shimmer 1.5s infinite; border-radius: var(--radius-sm); }
.evo-skeleton--text { height: 16px; margin-bottom: 8px; width: 80%; }
.evo-skeleton--title { height: 24px; margin-bottom: 12px; width: 60%; }
.evo-skeleton--image { height: 200px; width: 100%; border-radius: var(--radius-md); }
.evo-skeleton--avatar { height: 48px; width: 48px; border-radius: 50%; }
.evo-skeleton--card { padding: var(--space-lg); }
@keyframes evo-shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }"""
        html = """<div class="evo-skeleton evo-skeleton--card">
  <div class="evo-skeleton evo-skeleton--image"></div>
  <div style="padding-top:16px;">
    <div class="evo-skeleton evo-skeleton--title"></div>
    <div class="evo-skeleton evo-skeleton--text"></div>
    <div class="evo-skeleton evo-skeleton--text" style="width:60%;"></div>
  </div>
</div>"""
        return {"html": html, "css": css, "js": ""}

    def _gen_pagination(self, site_slug: str, config: Dict,
                        variant: str) -> Dict:
        html = """<nav class="evo-pagination" aria-label="Pagination">
  <a href="#" class="evo-pagination__link evo-pagination__prev">&laquo; Previous</a>
  <a href="#" class="evo-pagination__link evo-pagination__active">1</a>
  <a href="#" class="evo-pagination__link">2</a>
  <a href="#" class="evo-pagination__link">3</a>
  <span class="evo-pagination__ellipsis">&hellip;</span>
  <a href="#" class="evo-pagination__link">10</a>
  <a href="#" class="evo-pagination__link evo-pagination__next">Next &raquo;</a>
</nav>"""
        css = """.evo-pagination { display: flex; justify-content: center; align-items: center; gap: var(--space-xs); margin: var(--space-2xl) 0; flex-wrap: wrap; }
.evo-pagination__link { padding: 8px 14px; border: 1px solid var(--color-divider, #e5e7eb); border-radius: var(--radius-sm); font-size: var(--font-size-sm); color: var(--color-text); transition: all 0.2s; }
.evo-pagination__link:hover { background: var(--color-primary); color: #fff; border-color: var(--color-primary); }
.evo-pagination__active { background: var(--color-primary); color: #fff; border-color: var(--color-primary); }
.evo-pagination__ellipsis { padding: 8px 4px; color: var(--color-text-muted); }
.evo-pagination__prev, .evo-pagination__next { font-weight: 600; }"""
        return {"html": html, "css": css, "js": ""}

    def _gen_testimonials(self, site_slug: str, config: Dict,
                          variant: str) -> Dict:
        brand = config.get("name", site_slug)
        html = f"""<section class="evo-testimonials">
  <h2 class="evo-section__title">What Our Readers Say</h2>
  <div class="evo-grid evo-grid--3">
    <blockquote class="evo-testimonial">
      <div class="evo-testimonial__stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div>
      <p class="evo-testimonial__quote">"The best resource I've found. Incredibly helpful and well-researched content."</p>
      <cite class="evo-testimonial__author">— Happy Reader</cite>
    </blockquote>
    <blockquote class="evo-testimonial">
      <div class="evo-testimonial__stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div>
      <p class="evo-testimonial__quote">"I recommend {brand} to everyone. Their guides have saved me so much time."</p>
      <cite class="evo-testimonial__author">— Satisfied Subscriber</cite>
    </blockquote>
    <blockquote class="evo-testimonial">
      <div class="evo-testimonial__stars">&#9733;&#9733;&#9733;&#9733;&#9734;</div>
      <p class="evo-testimonial__quote">"Comprehensive, practical, and always up to date. My go-to site."</p>
      <cite class="evo-testimonial__author">— Loyal Reader</cite>
    </blockquote>
  </div>
</section>"""
        css = """.evo-testimonial { background: var(--color-bg); padding: var(--space-xl); border-radius: var(--radius-md); border: 1px solid var(--color-divider, #e5e7eb); margin: 0; }
.evo-testimonial__stars { color: #F59E0B; font-size: 18px; margin-bottom: var(--space-sm); }
.evo-testimonial__quote { font-style: italic; line-height: 1.7; color: var(--color-text); margin-bottom: var(--space-md); }
.evo-testimonial__author { font-style: normal; font-weight: 600; font-size: var(--font-size-sm); color: var(--color-text-muted); }"""
        return {"html": html, "css": css, "js": ""}

    def _gen_timeline(self, site_slug: str, config: Dict,
                      variant: str) -> Dict:
        html = """<div class="evo-timeline">
  <div class="evo-timeline__item">
    <div class="evo-timeline__dot"></div>
    <div class="evo-timeline__content">
      <h4>Step 1</h4>
      <p>First step description goes here.</p>
    </div>
  </div>
  <div class="evo-timeline__item">
    <div class="evo-timeline__dot"></div>
    <div class="evo-timeline__content">
      <h4>Step 2</h4>
      <p>Second step description goes here.</p>
    </div>
  </div>
  <div class="evo-timeline__item">
    <div class="evo-timeline__dot"></div>
    <div class="evo-timeline__content">
      <h4>Step 3</h4>
      <p>Third step description goes here.</p>
    </div>
  </div>
</div>"""
        css = """.evo-timeline { position: relative; padding-left: 40px; margin: var(--space-2xl) 0; }
.evo-timeline::before { content: ''; position: absolute; left: 15px; top: 0; bottom: 0; width: 2px; background: var(--color-primary); }
.evo-timeline__item { position: relative; margin-bottom: var(--space-xl); }
.evo-timeline__dot { position: absolute; left: -33px; top: 4px; width: 12px; height: 12px; border-radius: 50%; background: var(--color-primary); border: 3px solid var(--color-bg, #fff); }
.evo-timeline__content { background: var(--color-bg-alt); padding: var(--space-lg); border-radius: var(--radius-md); }
.evo-timeline__content h4 { margin-bottom: var(--space-xs); color: var(--color-primary); }"""
        return {"html": html, "css": css, "js": ""}

    def _gen_stat_counter(self, site_slug: str, config: Dict,
                          variant: str) -> Dict:
        html = """<section class="evo-stats">
  <div class="evo-grid evo-grid--4">
    <div class="evo-stat" data-target="1000" data-suffix="+">
      <span class="evo-stat__number">0</span>
      <span class="evo-stat__label">Articles Published</span>
    </div>
    <div class="evo-stat" data-target="50" data-suffix="k+">
      <span class="evo-stat__number">0</span>
      <span class="evo-stat__label">Monthly Readers</span>
    </div>
    <div class="evo-stat" data-target="4" data-suffix=".8/5">
      <span class="evo-stat__number">0</span>
      <span class="evo-stat__label">Reader Rating</span>
    </div>
    <div class="evo-stat" data-target="100" data-suffix="%">
      <span class="evo-stat__number">0</span>
      <span class="evo-stat__label">Free Content</span>
    </div>
  </div>
</section>"""
        css = """.evo-stat { text-align: center; padding: var(--space-xl); }
.evo-stat__number { font-family: var(--font-headline); font-size: var(--font-size-5xl); font-weight: 700; color: var(--color-primary); display: block; }
.evo-stat__label { font-size: var(--font-size-sm); color: var(--color-text-muted); margin-top: var(--space-xs); display: block; }"""
        js = """(function(){
  var observer = new IntersectionObserver(function(entries){
    entries.forEach(function(entry){
      if(!entry.isIntersecting) return;
      var el = entry.target;
      var target = parseInt(el.getAttribute('data-target')) || 0;
      var suffix = el.getAttribute('data-suffix') || '';
      var num = el.querySelector('.evo-stat__number');
      var current = 0;
      var step = Math.max(1, Math.ceil(target / 60));
      var interval = setInterval(function(){
        current += step;
        if(current >= target) { current = target; clearInterval(interval); }
        num.textContent = current + suffix;
      }, 20);
      observer.unobserve(el);
    });
  }, {threshold: 0.3});
  document.querySelectorAll('.evo-stat').forEach(function(s){ observer.observe(s); });
})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_social_share_bar(self, site_slug: str, config: Dict,
                              variant: str) -> Dict:
        html = """<div class="evo-share" id="evo-share">
  <span class="evo-share__label">Share:</span>
  <a class="evo-share__btn evo-share__btn--twitter" href="#" aria-label="Share on Twitter" data-share="twitter">𝕏</a>
  <a class="evo-share__btn evo-share__btn--facebook" href="#" aria-label="Share on Facebook" data-share="facebook">f</a>
  <a class="evo-share__btn evo-share__btn--linkedin" href="#" aria-label="Share on LinkedIn" data-share="linkedin">in</a>
  <a class="evo-share__btn evo-share__btn--pinterest" href="#" aria-label="Share on Pinterest" data-share="pinterest">P</a>
  <a class="evo-share__btn evo-share__btn--email" href="#" aria-label="Share via Email" data-share="email">&#9993;</a>
</div>"""
        css = """.evo-share { display: flex; align-items: center; gap: var(--space-sm); margin: var(--space-xl) 0; flex-wrap: wrap; }
.evo-share__label { font-size: var(--font-size-sm); font-weight: 600; color: var(--color-text-muted); }
.evo-share__btn { display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 50%; font-size: 14px; font-weight: 700; color: #fff; text-decoration: none; transition: transform 0.2s; }
.evo-share__btn:hover { transform: scale(1.1); color: #fff; }
.evo-share__btn--twitter { background: #000; }
.evo-share__btn--facebook { background: #1877F2; }
.evo-share__btn--linkedin { background: #0A66C2; }
.evo-share__btn--pinterest { background: #E60023; }
.evo-share__btn--email { background: #666; }"""
        js = """(function(){
  var url = encodeURIComponent(window.location.href);
  var title = encodeURIComponent(document.title);
  document.querySelectorAll('[data-share]').forEach(function(btn){
    btn.addEventListener('click', function(e){
      e.preventDefault();
      var t = this.getAttribute('data-share');
      var shareUrl = '';
      if(t==='twitter') shareUrl='https://twitter.com/intent/tweet?url='+url+'&text='+title;
      else if(t==='facebook') shareUrl='https://www.facebook.com/sharer/sharer.php?u='+url;
      else if(t==='linkedin') shareUrl='https://www.linkedin.com/sharing/share-offsite/?url='+url;
      else if(t==='pinterest') shareUrl='https://pinterest.com/pin/create/button/?url='+url+'&description='+title;
      else if(t==='email') shareUrl='mailto:?subject='+title+'&body='+url;
      if(shareUrl) window.open(shareUrl, '_blank', 'width=600,height=400');
    });
  });
})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_reading_progress_bar(self, site_slug: str, config: Dict,
                                   variant: str) -> Dict:
        css = """.evo-progress-bar { position: fixed; top: 0; left: 0; width: 0%; height: 3px; background: var(--color-primary); z-index: 99999; transition: width 0.1s linear; }
body.admin-bar .evo-progress-bar { top: 32px; }"""
        html = """<div class="evo-progress-bar" id="evo-progress-bar"></div>"""
        js = """(function(){
  var bar = document.getElementById('evo-progress-bar');
  if(!bar || !document.querySelector('article, .entry-content')) return;
  window.addEventListener('scroll', function(){
    var h = document.documentElement.scrollHeight - window.innerHeight;
    var pct = h > 0 ? (window.scrollY / h) * 100 : 0;
    bar.style.width = Math.min(100, pct) + '%';
  });
})();"""
        return {"html": html, "css": css, "js": js, "location": "site_wide_header"}

    def _gen_back_to_top(self, site_slug: str, config: Dict,
                         variant: str) -> Dict:
        html = """<button id="evo-back-to-top" class="evo-back-to-top" aria-label="Back to top" style="display:none;">&#8679;</button>"""
        css = """.evo-back-to-top { position: fixed; bottom: 24px; right: 24px; z-index: 9997; width: 44px; height: 44px; border-radius: 50%; background: var(--color-primary); color: #fff; border: none; font-size: 22px; cursor: pointer; box-shadow: var(--shadow-md); transition: opacity 0.3s, transform 0.3s; }
.evo-back-to-top:hover { transform: scale(1.1); }
@media (max-width: 768px) { .evo-back-to-top { bottom: 16px; right: 16px; } }"""
        js = """(function(){
  var btn = document.getElementById('evo-back-to-top');
  if(!btn) return;
  window.addEventListener('scroll', function(){
    btn.style.display = window.scrollY > 400 ? 'block' : 'none';
  });
  btn.addEventListener('click', function(){
    window.scrollTo({top: 0, behavior: 'smooth'});
  });
})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_cookie_consent(self, site_slug: str, config: Dict,
                            variant: str) -> Dict:
        brand = config.get("name", site_slug)
        html = f"""<div id="evo-cookie-consent" class="evo-cookie-consent" style="display:none;">
  <div class="evo-cookie-consent__inner">
    <p>{brand} uses cookies to improve your experience. By continuing, you agree to our <a href="/cookie-policy">Cookie Policy</a>.</p>
    <div class="evo-cookie-consent__actions">
      <button id="evo-cookie-accept" class="evo-btn evo-btn-primary" style="padding:8px 24px;">Accept</button>
      <button id="evo-cookie-decline" class="evo-btn evo-btn-secondary" style="padding:8px 16px;">Decline</button>
    </div>
  </div>
</div>"""
        css = """.evo-cookie-consent { position: fixed; bottom: 0; left: 0; right: 0; z-index: 99998; background: var(--color-bg, #fff); padding: 16px 24px; box-shadow: 0 -4px 20px rgba(0,0,0,0.1); border-top: 1px solid var(--color-divider, #e5e7eb); }
.evo-cookie-consent__inner { max-width: var(--max-width, 1200px); margin: 0 auto; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.evo-cookie-consent__inner p { margin: 0; font-size: var(--font-size-sm); flex: 1; min-width: 250px; }
.evo-cookie-consent__actions { display: flex; gap: 8px; }"""
        js = """(function(){
  var banner = document.getElementById('evo-cookie-consent');
  if(!banner) return;
  if(localStorage.getItem('evo_cookie_consent')) return;
  banner.style.display = 'block';
  document.getElementById('evo-cookie-accept').addEventListener('click', function(){
    localStorage.setItem('evo_cookie_consent', 'accepted');
    banner.style.display = 'none';
  });
  document.getElementById('evo-cookie-decline').addEventListener('click', function(){
    localStorage.setItem('evo_cookie_consent', 'declined');
    banner.style.display = 'none';
  });
})();"""
        return {"html": html, "css": css, "js": js}

    def _gen_dark_mode_toggle(self, site_slug: str, config: Dict,
                              variant: str) -> Dict:
        html = """<button id="evo-dark-toggle" class="evo-dark-toggle" aria-label="Toggle dark mode" title="Toggle dark mode">
  <span class="evo-dark-toggle__icon evo-dark-toggle__sun">&#9728;</span>
  <span class="evo-dark-toggle__icon evo-dark-toggle__moon" style="display:none;">&#9790;</span>
</button>"""
        css = """.evo-dark-toggle { position: fixed; bottom: 80px; right: 24px; z-index: 9996; width: 44px; height: 44px; border-radius: 50%; background: var(--color-bg-alt); border: 1px solid var(--color-divider, #e5e7eb); cursor: pointer; font-size: 20px; display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-sm); transition: background 0.3s; }
.evo-dark-toggle:hover { box-shadow: var(--shadow-md); }
[data-theme="dark"] .evo-dark-toggle { background: #1E293B; border-color: #334155; }"""
        js = """(function(){
  var toggle = document.getElementById('evo-dark-toggle');
  if(!toggle) return;
  var sun = toggle.querySelector('.evo-dark-toggle__sun');
  var moon = toggle.querySelector('.evo-dark-toggle__moon');

  function setTheme(dark) {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    sun.style.display = dark ? 'none' : 'inline';
    moon.style.display = dark ? 'inline' : 'none';
    localStorage.setItem('evo_theme', dark ? 'dark' : 'light');
  }

  // Load saved preference or system preference
  var saved = localStorage.getItem('evo_theme');
  if (saved) { setTheme(saved === 'dark'); }
  else if (window.matchMedia('(prefers-color-scheme: dark)').matches) { setTheme(true); }

  toggle.addEventListener('click', function() {
    var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    setTheme(!isDark);
  });
})();"""
        return {"html": html, "css": css, "js": js}
