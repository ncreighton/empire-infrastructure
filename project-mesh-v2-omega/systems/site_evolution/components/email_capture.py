"""
Email Capture System — Exit intent, scroll trigger, timed popup, inline, slide-in.
All variants integrate with Systeme.io forms and deploy as Code Snippets.
"""

import logging
from typing import Dict

from systems.site_evolution.utils import load_site_config, get_site_brand_name

log = logging.getLogger(__name__)


class EmailCaptureSystem:
    """Generate email capture components with multiple trigger types."""

    def generate_exit_intent_popup(self, site_slug: str) -> Dict:
        """Popup triggered when mouse moves toward browser close/back button."""
        config = load_site_config(site_slug)
        brand = get_site_brand_name(site_slug)
        colors = config.get("brand", {}).get("colors", {})
        primary = colors.get("primary", "#6B46C1")

        html = f"""<!-- Exit Intent Popup -->
<div id="evo-exit-popup" class="evo-popup" style="display:none;">
  <div class="evo-popup__backdrop"></div>
  <div class="evo-popup__content">
    <button class="evo-popup__close" aria-label="Close">&times;</button>
    <h2 class="evo-popup__title">Wait! Don't miss out</h2>
    <p class="evo-popup__text">Get our best content delivered straight to your inbox. Join the {brand} community today.</p>
    <form class="evo-popup__form" action="#" method="post">
      <input type="email" name="email" placeholder="Enter your email" required class="evo-popup__input">
      <button type="submit" class="evo-btn evo-btn-primary">Yes, Send Me Updates</button>
    </form>
    <p class="evo-popup__disclaimer">No spam. Unsubscribe anytime.</p>
  </div>
</div>"""

        css = f""".evo-popup {{ position: fixed; inset: 0; z-index: 99999; display: flex; align-items: center; justify-content: center; }}
.evo-popup__backdrop {{ position: absolute; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); }}
.evo-popup__content {{ position: relative; background: #fff; border-radius: 16px; padding: 48px 40px; max-width: 480px; width: 90%; text-align: center; animation: evo-popIn 0.3s ease; }}
.evo-popup__close {{ position: absolute; top: 12px; right: 16px; background: none; border: none; font-size: 28px; cursor: pointer; color: #666; line-height: 1; }}
.evo-popup__title {{ font-size: 24px; font-weight: 700; margin-bottom: 12px; color: #1a1a1a; }}
.evo-popup__text {{ color: #666; margin-bottom: 24px; line-height: 1.6; }}
.evo-popup__form {{ display: flex; flex-direction: column; gap: 12px; }}
.evo-popup__input {{ padding: 14px 16px; border: 2px solid #e5e7eb; border-radius: 8px; font-size: 16px; }}
.evo-popup__input:focus {{ border-color: {primary}; outline: none; }}
.evo-popup__disclaimer {{ font-size: 12px; color: #999; margin-top: 12px; }}
@keyframes evo-popIn {{ from {{ opacity: 0; transform: scale(0.9); }} to {{ opacity: 1; transform: scale(1); }} }}
@media (max-width: 480px) {{ .evo-popup__content {{ padding: 32px 20px; }} }}"""

        js = """(function(){
  var popup = document.getElementById('evo-exit-popup');
  if (!popup) return;
  var shown = localStorage.getItem('evo_exit_shown');
  if (shown) return;

  document.addEventListener('mouseleave', function(e) {
    if (e.clientY < 10) {
      popup.style.display = 'flex';
      localStorage.setItem('evo_exit_shown', Date.now());
    }
  });

  popup.querySelector('.evo-popup__close').addEventListener('click', function() {
    popup.style.display = 'none';
  });
  popup.querySelector('.evo-popup__backdrop').addEventListener('click', function() {
    popup.style.display = 'none';
  });
})();"""
        return {"html": html, "css": css, "js": js}

    def generate_scroll_trigger(self, site_slug: str, threshold: int = 50) -> Dict:
        """Popup shown after user scrolls past threshold percentage."""
        config = load_site_config(site_slug)
        brand = get_site_brand_name(site_slug)

        js = f"""(function(){{
  var triggered = sessionStorage.getItem('evo_scroll_shown');
  if (triggered) return;
  var threshold = {threshold};

  window.addEventListener('scroll', function() {{
    var scrollPct = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
    if (scrollPct >= threshold) {{
      var popup = document.getElementById('evo-exit-popup');
      if (popup && popup.style.display === 'none') {{
        popup.style.display = 'flex';
        sessionStorage.setItem('evo_scroll_shown', '1');
      }}
    }}
  }});
}})();"""
        return {"html": "", "css": "", "js": js}

    def generate_timed_popup(self, site_slug: str, delay_seconds: int = 30) -> Dict:
        """Popup shown after delay_seconds on page."""
        js = f"""(function(){{
  var shown = sessionStorage.getItem('evo_timed_shown');
  if (shown) return;
  setTimeout(function() {{
    var popup = document.getElementById('evo-exit-popup');
    if (popup && popup.style.display === 'none') {{
      popup.style.display = 'flex';
      sessionStorage.setItem('evo_timed_shown', '1');
    }}
  }}, {delay_seconds * 1000});
}})();"""
        return {"html": "", "css": "", "js": js}

    def generate_inline_capture(self, site_slug: str) -> Dict:
        """In-content email capture box (inserted after 2nd paragraph)."""
        config = load_site_config(site_slug)
        brand = get_site_brand_name(site_slug)
        colors = config.get("brand", {}).get("colors", {})
        primary = colors.get("primary", "#6B46C1")

        html = f"""<div class="evo-inline-capture">
  <div class="evo-inline-capture__icon">&#9993;</div>
  <div class="evo-inline-capture__text">
    <strong>Get {brand} in your inbox</strong>
    <span>Free weekly insights from our expert team.</span>
  </div>
  <form class="evo-inline-capture__form" action="#" method="post">
    <input type="email" name="email" placeholder="Email address" required>
    <button type="submit" class="evo-btn evo-btn-primary">Subscribe</button>
  </form>
</div>"""

        css = f""".evo-inline-capture {{ display: flex; align-items: center; gap: 16px; padding: 20px 24px; background: linear-gradient(135deg, {primary}08, {primary}15); border: 1px solid {primary}30; border-radius: 12px; margin: 32px 0; flex-wrap: wrap; }}
.evo-inline-capture__icon {{ font-size: 28px; }}
.evo-inline-capture__text {{ flex: 1; min-width: 200px; }}
.evo-inline-capture__text strong {{ display: block; font-size: 16px; }}
.evo-inline-capture__text span {{ font-size: 14px; color: #666; }}
.evo-inline-capture__form {{ display: flex; gap: 8px; }}
.evo-inline-capture__form input {{ padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; min-width: 200px; }}
@media (max-width: 640px) {{ .evo-inline-capture {{ flex-direction: column; text-align: center; }} .evo-inline-capture__form {{ width: 100%; flex-direction: column; }} .evo-inline-capture__form input {{ min-width: auto; width: 100%; }} }}"""

        return {"html": html, "css": css, "js": ""}

    def generate_slide_in(self, site_slug: str) -> Dict:
        """Bottom-right slide-in CTA that appears after scrolling."""
        config = load_site_config(site_slug)
        brand = get_site_brand_name(site_slug)
        colors = config.get("brand", {}).get("colors", {})
        primary = colors.get("primary", "#6B46C1")

        html = f"""<div id="evo-slide-in" class="evo-slide-in" style="display:none;">
  <button class="evo-slide-in__close" aria-label="Close">&times;</button>
  <strong>Join {brand}</strong>
  <p>Get expert insights free every week.</p>
  <form action="#" method="post" class="evo-slide-in__form">
    <input type="email" name="email" placeholder="Email" required>
    <button type="submit" class="evo-btn evo-btn-primary" style="width:100%;">Subscribe</button>
  </form>
</div>"""

        css = f""".evo-slide-in {{ position: fixed; bottom: 20px; right: 20px; z-index: 9998; background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.15); max-width: 320px; animation: evo-slideUp 0.4s ease; }}
.evo-slide-in__close {{ position: absolute; top: 8px; right: 12px; background: none; border: none; font-size: 20px; cursor: pointer; color: #999; }}
.evo-slide-in p {{ font-size: 14px; color: #666; margin: 8px 0 16px; }}
.evo-slide-in__form input {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 8px; }}
@keyframes evo-slideUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@media (max-width: 480px) {{ .evo-slide-in {{ right: 10px; left: 10px; max-width: none; bottom: 10px; }} }}"""

        js = """(function(){
  var el = document.getElementById('evo-slide-in');
  if (!el) return;
  var dismissed = sessionStorage.getItem('evo_slidein_dismissed');
  if (dismissed) return;

  window.addEventListener('scroll', function() {
    if (window.scrollY > 800 && el.style.display === 'none') {
      el.style.display = 'block';
    }
  });

  el.querySelector('.evo-slide-in__close').addEventListener('click', function() {
    el.style.display = 'none';
    sessionStorage.setItem('evo_slidein_dismissed', '1');
  });
})();"""
        return {"html": html, "css": css, "js": js}

    def generate_capture_snippet(self, site_slug: str, trigger: str = "exit") -> Dict:
        """Generate a complete email capture snippet ready for deployment.

        Args:
            trigger: "exit", "scroll", "timed", "inline", "slide_in"
        """
        if trigger == "exit":
            return self.generate_exit_intent_popup(site_slug)
        elif trigger == "scroll":
            popup = self.generate_exit_intent_popup(site_slug)
            scroll = self.generate_scroll_trigger(site_slug)
            popup["js"] = scroll["js"]  # Replace exit-intent JS with scroll trigger
            return popup
        elif trigger == "timed":
            popup = self.generate_exit_intent_popup(site_slug)
            timed = self.generate_timed_popup(site_slug)
            popup["js"] = timed["js"]
            return popup
        elif trigger == "inline":
            return self.generate_inline_capture(site_slug)
        elif trigger == "slide_in":
            return self.generate_slide_in(site_slug)
        else:
            return self.generate_exit_intent_popup(site_slug)
