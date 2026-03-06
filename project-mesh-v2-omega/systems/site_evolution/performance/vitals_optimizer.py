"""
Vitals Optimizer — Core Web Vitals enhancement.
Generates performance snippets: critical CSS, lazy loading, preload hints,
font optimization. All outputs are WPCode-ready.
"""

import logging
from typing import Dict, List

from systems.site_evolution.utils import load_site_config

log = logging.getLogger(__name__)


class VitalsOptimizer:
    """Generate performance optimization snippets for WordPress."""

    def generate_critical_css(self, site_slug: str) -> str:
        """Generate above-the-fold critical CSS for inline in <head>."""
        return """/* Critical CSS — Above the fold */
*,*::before,*::after{box-sizing:border-box}
body{margin:0;font-family:var(--font-body,system-ui,-apple-system,sans-serif);
line-height:var(--line-height-body,1.7);color:var(--color-text,#1a1a1a);
background:var(--color-bg,#fff);-webkit-font-smoothing:antialiased}
h1,h2,h3{font-family:var(--font-headline,inherit);line-height:1.2;margin:0 0 0.5em}
img{max-width:100%;height:auto;display:block}
a{color:var(--color-primary,#3b82f6);text-decoration:none}
.evo-nav{position:sticky;top:0;z-index:1000;display:flex;align-items:center;
justify-content:space-between;padding:0 1.5rem;height:70px;
background:var(--color-bg,#fff);box-shadow:0 1px 2px rgba(0,0,0,0.06)}
.evo-hero{display:flex;flex-direction:column;align-items:center;
justify-content:center;text-align:center;min-height:70vh;
padding:4rem 1.5rem;color:#fff}"""

    def generate_lazy_load_snippet(self, site_slug: str) -> str:
        """Generate native lazy loading for images + iframes."""
        return """<?php
/**
 * Add native lazy loading to images and iframes
 */
function evo_lazy_load_content($content) {
    if (is_admin() || is_feed()) return $content;

    // Add loading="lazy" to images without it
    $content = preg_replace(
        '/<img(?![^>]*loading=)([^>]*)>/i',
        '<img loading="lazy"$1>',
        $content
    );

    // Add loading="lazy" to iframes
    $content = preg_replace(
        '/<iframe(?![^>]*loading=)([^>]*)>/i',
        '<iframe loading="lazy"$1>',
        $content
    );

    return $content;
}
add_filter('the_content', 'evo_lazy_load_content', 99);

/**
 * Add fetchpriority="high" to above-the-fold hero images
 */
function evo_hero_image_priority($content) {
    // First image in content gets high priority
    $content = preg_replace(
        '/<img([^>]*?)loading="lazy"(.*?)>/i',
        '<img$1fetchpriority="high"$2>',
        $content,
        1  // Only first image
    );
    return $content;
}
add_filter('the_content', 'evo_hero_image_priority', 100);
"""

    def generate_preload_hints(self, site_slug: str) -> str:
        """Generate <link rel="preload"> for fonts, hero images, critical CSS."""
        config = load_site_config(site_slug)
        fonts = config.get("brand", {}).get("fonts", {})
        headline_font = fonts.get("headline", "Inter")
        body_font = fonts.get("body", "Inter")

        headline_safe = headline_font.replace(" ", "+")
        body_safe = body_font.replace(" ", "+")

        return f"""<?php
function evo_preload_hints() {{
    // Preload critical fonts
    echo '<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>' . "\\n";
    echo '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>' . "\\n";

    // DNS prefetch for common services
    echo '<link rel="dns-prefetch" href="//www.googletagmanager.com">' . "\\n";
    echo '<link rel="dns-prefetch" href="//www.google-analytics.com">' . "\\n";
    echo '<link rel="dns-prefetch" href="//pagead2.googlesyndication.com">' . "\\n";
    echo '<link rel="dns-prefetch" href="//cdn.jsdelivr.net">' . "\\n";

    // Preload font stylesheet
    echo '<link rel="preload" href="https://fonts.googleapis.com/css2?family={headline_safe}:wght@700&family={body_safe}:wght@400&display=swap" as="style">' . "\\n";
}}
add_action('wp_head', 'evo_preload_hints', 1);
"""

    def generate_font_optimization(self, site_slug: str) -> str:
        """Generate font-display: swap and subset loading."""
        return """<?php
/**
 * Optimize font loading with font-display: swap
 */
function evo_optimize_font_loading($html) {
    // Add font-display=swap to Google Fonts URLs
    $html = str_replace(
        'fonts.googleapis.com/css2?',
        'fonts.googleapis.com/css2?display=swap&',
        $html
    );
    return $html;
}
add_filter('style_loader_tag', 'evo_optimize_font_loading', 10, 1);

/**
 * Add font-display: swap to inline font-face declarations
 */
function evo_font_display_swap() {
    echo '<style>@font-face{font-display:swap!important}</style>' . "\\n";
}
add_action('wp_head', 'evo_font_display_swap', 1);
"""

    def generate_image_optimization_snippet(self) -> str:
        """Generate WebP conversion hints and srcset optimization."""
        return """<?php
/**
 * Add decoding="async" to all images for non-blocking decode
 */
function evo_async_images($content) {
    if (is_admin()) return $content;
    $content = preg_replace(
        '/<img(?![^>]*decoding=)([^>]*)>/i',
        '<img decoding="async"$1>',
        $content
    );
    return $content;
}
add_filter('the_content', 'evo_async_images', 98);

/**
 * Set default image quality for WordPress-generated sizes
 */
function evo_image_quality($quality) {
    return 82;  // Good balance of quality vs size
}
add_filter('jpeg_quality', 'evo_image_quality');
add_filter('wp_editor_set_quality', 'evo_image_quality');
"""

    def generate_caching_headers(self) -> str:
        """Generate browser caching rules (LiteSpeed-compatible)."""
        return """<?php
/**
 * Add cache-control headers for static assets
 */
function evo_cache_headers() {
    if (is_admin()) return;

    // Don't cache dynamic pages for logged-in users
    if (is_user_logged_in()) return;

    // Add cache headers for front-end pages
    header('Cache-Control: public, max-age=3600, s-maxage=86400');
    header('Vary: Accept-Encoding');
}
add_action('send_headers', 'evo_cache_headers');
"""

    def generate_cls_prevention(self) -> str:
        """Generate CLS (Cumulative Layout Shift) prevention snippet."""
        return """<?php
/**
 * Prevent CLS by reserving space for common elements
 */
function evo_cls_prevention() {
    echo '<style>
/* Reserve space for ads to prevent layout shift */
.adsbygoogle { min-height: 250px; }
/* Reserve space for images */
img:not([width]):not([height]) { aspect-ratio: 16/9; }
/* Prevent font-swap layout shift */
body { font-synthesis: none; }
</style>' . "\\n";
}
add_action('wp_head', 'evo_cls_prevention', 2);
"""

    def generate_script_defer(self) -> str:
        """Defer non-critical JavaScript loading."""
        return """<?php
/**
 * Add defer attribute to non-critical scripts
 */
function evo_defer_scripts($tag, $handle, $src) {
    // Don't defer admin or critical scripts
    $no_defer = array('jquery-core', 'jquery-migrate', 'wp-embed');
    if (in_array($handle, $no_defer)) return $tag;
    if (is_admin()) return $tag;

    // Add defer to all other scripts
    if (strpos($tag, 'defer') === false && strpos($tag, '<script') !== false) {
        $tag = str_replace(' src=', ' defer src=', $tag);
    }
    return $tag;
}
add_filter('script_loader_tag', 'evo_defer_scripts', 10, 3);
"""

    def generate_all_performance_snippets(self, site_slug: str) -> Dict[str, str]:
        """Generate all performance optimization snippets."""
        return {
            "critical_css": self.generate_critical_css(site_slug),
            "lazy_load": self.generate_lazy_load_snippet(site_slug),
            "preload_hints": self.generate_preload_hints(site_slug),
            "font_optimization": self.generate_font_optimization(site_slug),
            "image_optimization": self.generate_image_optimization_snippet(),
            "caching_headers": self.generate_caching_headers(),
            "cls_prevention": self.generate_cls_prevention(),
            "script_defer": self.generate_script_defer(),
        }
