"""
CSS Engine — Generates complete CSS frameworks from design tokens.
Produces 700-1200 lines of production CSS per site.
"""

import logging
from typing import Dict

from systems.site_evolution.designer.design_generator import DesignSystem, DARK_MODE_SITES

log = logging.getLogger(__name__)


class CSSEngine:
    """Generate full CSS frameworks from DesignSystem objects."""

    def generate_root_variables(self, ds: DesignSystem) -> str:
        """Generate :root CSS custom properties block."""
        lines = [":root {"]
        for name, value in sorted(ds.css_variables.items()):
            lines.append(f"  {name}: {value};")
        lines.append("}")
        return "\n".join(lines)

    def generate_typography_css(self, ds: DesignSystem) -> str:
        """Generate font imports + typography styles."""
        fonts = set()
        for role in ("headline", "subhead", "body", "accent", "badge"):
            info = ds.typography_stack.get(role, {})
            family = info.get("family", "Inter")
            weight = info.get("weight", 400)
            fonts.add((family, weight))

        # Google Fonts import
        imports = []
        for family, weight in sorted(fonts):
            safe_family = family.replace(" ", "+")
            imports.append(
                f"@import url('https://fonts.googleapis.com/css2?"
                f"family={safe_family}:wght@{weight}&display=swap');"
            )

        css = "\n".join(imports) + "\n\n"
        css += """/* Typography */
body {
  font-family: var(--font-body);
  font-weight: var(--font-weight-body);
  font-size: var(--font-size-base);
  line-height: var(--line-height-body);
  letter-spacing: var(--letter-spacing-body);
  color: var(--color-text);
  background-color: var(--color-bg);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-headline);
  font-weight: var(--font-weight-headline);
  line-height: var(--line-height-headline);
  letter-spacing: var(--letter-spacing-headline);
  color: var(--color-text);
  margin-bottom: var(--space-md);
}

h1 { font-size: var(--font-size-5xl); }
h2 { font-size: var(--font-size-4xl); }
h3 { font-size: var(--font-size-3xl); }
h4 { font-size: var(--font-size-2xl); }
h5 { font-size: var(--font-size-xl); }
h6 { font-size: var(--font-size-lg); }

p { margin-bottom: var(--space-md); }

a {
  color: var(--color-primary);
  text-decoration: none;
  transition: color var(--transition-fast);
}
a:hover { color: var(--color-accent); }

.text-muted { color: var(--color-text-muted); }
.text-primary { color: var(--color-primary); }
.text-accent { color: var(--color-accent); }

blockquote {
  border-left: 4px solid var(--color-primary);
  padding: var(--space-md) var(--space-lg);
  margin: var(--space-xl) 0;
  background: var(--color-bg-alt);
  font-style: italic;
}

code {
  background: var(--color-bg-alt);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
}

pre code {
  display: block;
  padding: var(--space-lg);
  overflow-x: auto;
}
"""
        return css

    def generate_component_css(self, ds: DesignSystem) -> str:
        """Generate CSS for all component types."""
        return """/* -- Components -- */

/* Badge */
.badge {
  display: inline-block;
  padding: 4px 12px;
  font-size: var(--font-size-xs);
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  background: var(--color-badge-bg);
  color: var(--color-badge-text);
  border-radius: var(--radius-full);
}

/* Card */
.evo-card {
  background: var(--color-bg);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  transition: transform var(--transition-base), box-shadow var(--transition-base);
}
.evo-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}
.evo-card__image {
  width: 100%;
  aspect-ratio: 16/9;
  object-fit: cover;
}
.evo-card__body {
  padding: var(--space-lg);
}
.evo-card__title {
  font-family: var(--font-headline);
  font-size: var(--font-size-xl);
  margin-bottom: var(--space-sm);
}
.evo-card__excerpt {
  color: var(--color-text-muted);
  font-size: var(--font-size-sm);
  line-height: 1.6;
}
.evo-card__meta {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-top: var(--space-md);
  font-size: var(--font-size-xs);
  color: var(--color-text-muted);
}

/* Button */
.evo-btn {
  display: inline-flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 12px 28px;
  font-family: var(--font-body);
  font-weight: 600;
  font-size: var(--font-size-sm);
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
  text-decoration: none;
}
.evo-btn-primary {
  background: var(--color-primary);
  color: #fff;
}
.evo-btn-primary:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  color: #fff;
}
.evo-btn-secondary {
  background: transparent;
  color: var(--color-primary);
  border: 2px solid var(--color-primary);
}
.evo-btn-secondary:hover {
  background: var(--color-primary);
  color: #fff;
}

/* Navigation */
.evo-nav {
  position: sticky;
  top: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-xl);
  height: 70px;
  background: var(--color-bg);
  backdrop-filter: blur(12px);
  box-shadow: var(--shadow-sm);
}
.evo-nav__logo {
  font-family: var(--font-headline);
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-headline);
  color: var(--color-primary);
}
.evo-nav__links {
  display: flex;
  align-items: center;
  gap: var(--space-lg);
  list-style: none;
  margin: 0;
  padding: 0;
}
.evo-nav__links a {
  font-size: var(--font-size-sm);
  font-weight: 500;
  color: var(--color-text);
}
.evo-nav__links a:hover { color: var(--color-primary); }

/* Hero */
.evo-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  min-height: 70vh;
  padding: var(--space-section) var(--space-xl);
  background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
  color: #fff;
}
.evo-hero__headline {
  font-size: clamp(2rem, 5vw, var(--font-size-5xl));
  max-width: 800px;
  margin-bottom: var(--space-lg);
  color: #fff;
}
.evo-hero__subheadline {
  font-size: var(--font-size-xl);
  opacity: 0.9;
  max-width: 600px;
  margin-bottom: var(--space-2xl);
}
.evo-hero__ctas {
  display: flex;
  gap: var(--space-md);
  flex-wrap: wrap;
  justify-content: center;
}

/* Section */
.evo-section {
  padding: var(--space-section) var(--space-xl);
  max-width: var(--max-width);
  margin: 0 auto;
}
.evo-section--alt { background: var(--color-bg-alt); }
.evo-section__title {
  text-align: center;
  margin-bottom: var(--space-3xl);
}

/* Grid */
.evo-grid {
  display: grid;
  gap: var(--space-xl);
}
.evo-grid--2 { grid-template-columns: repeat(2, 1fr); }
.evo-grid--3 { grid-template-columns: repeat(3, 1fr); }
.evo-grid--4 { grid-template-columns: repeat(4, 1fr); }

/* Newsletter */
.evo-newsletter {
  padding: var(--space-section) var(--space-xl);
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: #fff;
  text-align: center;
  border-radius: var(--radius-lg);
}
.evo-newsletter__form {
  display: flex;
  gap: var(--space-sm);
  max-width: 500px;
  margin: var(--space-xl) auto 0;
}
.evo-newsletter__input {
  flex: 1;
  padding: 14px 20px;
  border: none;
  border-radius: var(--radius-md);
  font-size: var(--font-size-base);
}
.evo-newsletter__input:focus { outline: 2px solid #fff; }

/* Author Box */
.evo-author {
  display: flex;
  gap: var(--space-lg);
  padding: var(--space-xl);
  background: var(--color-bg-alt);
  border-radius: var(--radius-md);
  margin: var(--space-2xl) 0;
}
.evo-author__avatar {
  width: 80px;
  height: 80px;
  border-radius: var(--radius-full);
  object-fit: cover;
}
.evo-author__name {
  font-family: var(--font-headline);
  font-weight: 600;
  margin-bottom: var(--space-xs);
}
.evo-author__bio {
  font-size: var(--font-size-sm);
  color: var(--color-text-muted);
}

/* Table of Contents */
.evo-toc {
  background: var(--color-bg-alt);
  padding: var(--space-lg);
  border-radius: var(--radius-md);
  border-left: 4px solid var(--color-primary);
  margin: var(--space-xl) 0;
}
.evo-toc__title {
  font-family: var(--font-headline);
  font-size: var(--font-size-lg);
  margin-bottom: var(--space-md);
}
.evo-toc ol {
  padding-left: var(--space-lg);
  margin: 0;
}
.evo-toc li { margin-bottom: var(--space-xs); }
.evo-toc a {
  color: var(--color-text-muted);
  font-size: var(--font-size-sm);
}
.evo-toc a:hover { color: var(--color-primary); }

/* Footer */
.evo-footer {
  padding: var(--space-section) var(--space-xl) var(--space-2xl);
  background: var(--color-bg-alt);
  border-top: 1px solid var(--color-divider);
}
.evo-footer__grid {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: var(--space-2xl);
  max-width: var(--max-width);
  margin: 0 auto;
}
.evo-footer__brand {
  font-family: var(--font-headline);
  font-size: var(--font-size-xl);
  color: var(--color-primary);
  margin-bottom: var(--space-md);
}
.evo-footer__links {
  list-style: none;
  padding: 0;
  margin: 0;
}
.evo-footer__links li { margin-bottom: var(--space-sm); }
.evo-footer__links a {
  color: var(--color-text-muted);
  font-size: var(--font-size-sm);
}
.evo-footer__links a:hover { color: var(--color-primary); }
.evo-footer__copyright {
  text-align: center;
  margin-top: var(--space-2xl);
  padding-top: var(--space-lg);
  border-top: 1px solid var(--color-divider);
  font-size: var(--font-size-xs);
  color: var(--color-text-muted);
}

/* Comparison Table */
.evo-comparison {
  width: 100%;
  border-collapse: collapse;
  margin: var(--space-xl) 0;
}
.evo-comparison th,
.evo-comparison td {
  padding: var(--space-md);
  text-align: left;
  border-bottom: 1px solid var(--color-divider);
}
.evo-comparison th {
  background: var(--color-bg-alt);
  font-family: var(--font-headline);
  font-size: var(--font-size-sm);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.evo-comparison tr:hover td { background: var(--color-bg-alt); }

/* CTA Sections */
.evo-cta {
  padding: var(--space-2xl);
  text-align: center;
  border-radius: var(--radius-lg);
  margin: var(--space-2xl) 0;
}
.evo-cta--gradient {
  background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
  color: #fff;
}
.evo-cta--subtle {
  background: var(--color-bg-alt);
  border: 2px solid var(--color-divider);
}

/* Breadcrumbs */
.evo-breadcrumbs {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md) 0;
  font-size: var(--font-size-sm);
  color: var(--color-text-muted);
}
.evo-breadcrumbs a { color: var(--color-text-muted); }
.evo-breadcrumbs a:hover { color: var(--color-primary); }
.evo-breadcrumbs__sep { opacity: 0.5; }
"""

    def generate_utility_classes(self, ds: DesignSystem) -> str:
        """Generate utility CSS classes."""
        return """/* -- Utilities -- */
.flex { display: flex; }
.flex-col { flex-direction: column; }
.flex-center { display: flex; align-items: center; justify-content: center; }
.flex-between { display: flex; align-items: center; justify-content: space-between; }
.flex-wrap { flex-wrap: wrap; }
.gap-sm { gap: var(--space-sm); }
.gap-md { gap: var(--space-md); }
.gap-lg { gap: var(--space-lg); }

.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }

.mx-auto { margin-left: auto; margin-right: auto; }
.mt-section { margin-top: var(--space-section); }
.mb-section { margin-bottom: var(--space-section); }
.p-md { padding: var(--space-md); }
.p-lg { padding: var(--space-lg); }
.p-xl { padding: var(--space-xl); }

.bg-primary { background-color: var(--color-primary); }
.bg-alt { background-color: var(--color-bg-alt); }
.bg-surface { background-color: var(--color-bg); }

.rounded { border-radius: var(--radius-md); }
.rounded-lg { border-radius: var(--radius-lg); }
.rounded-full { border-radius: var(--radius-full); }

.shadow-sm { box-shadow: var(--shadow-sm); }
.shadow-md { box-shadow: var(--shadow-md); }
.shadow-lg { box-shadow: var(--shadow-lg); }

.w-full { width: 100%; }
.max-w-content { max-width: var(--max-width-content); }
.max-w-page { max-width: var(--max-width); }

.sr-only {
  position: absolute; width: 1px; height: 1px;
  padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}

.visually-hidden { composes: sr-only; }
"""

    def generate_animation_css(self, ds: DesignSystem) -> str:
        """Generate keyframe animations."""
        return """/* -- Animations -- */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.animate-fadeIn { animation: fadeIn 0.6s ease forwards; }
.animate-slideUp { animation: slideUp 0.5s ease forwards; }
.animate-scaleIn { animation: scaleIn 0.4s ease forwards; }

/* Scroll-triggered animation (requires JS IntersectionObserver) */
[data-animate] {
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.6s ease, transform 0.6s ease;
}
[data-animate].visible {
  opacity: 1;
  transform: translateY(0);
}
"""

    def generate_responsive_css(self, ds: DesignSystem) -> str:
        """Generate responsive media queries."""
        return """/* -- Responsive -- */
@media (max-width: 1280px) {
  .evo-grid--4 { grid-template-columns: repeat(3, 1fr); }
}

@media (max-width: 1024px) {
  .evo-grid--3 { grid-template-columns: repeat(2, 1fr); }
  .evo-grid--4 { grid-template-columns: repeat(2, 1fr); }
  .evo-footer__grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 768px) {
  .evo-grid--2,
  .evo-grid--3,
  .evo-grid--4 { grid-template-columns: 1fr; }

  .evo-hero { min-height: 50vh; padding: var(--space-2xl) var(--space-md); }
  .evo-hero__headline { font-size: var(--font-size-3xl); }
  .evo-hero__ctas { flex-direction: column; align-items: center; }

  .evo-nav { padding: 0 var(--space-md); }
  .evo-nav__links { display: none; }

  .evo-section { padding: var(--space-2xl) var(--space-md); }

  .evo-footer__grid { grid-template-columns: 1fr; }

  .evo-newsletter__form { flex-direction: column; }

  .evo-author { flex-direction: column; align-items: center; text-align: center; }

  .evo-comparison { display: block; overflow-x: auto; }
}

@media (max-width: 480px) {
  h1 { font-size: var(--font-size-3xl); }
  h2 { font-size: var(--font-size-2xl); }
  h3 { font-size: var(--font-size-xl); }
}
"""

    def generate_dark_mode_css(self, ds: DesignSystem) -> str:
        """Generate dark mode variant (for supported sites)."""
        if not ds.supports_dark_mode:
            return ""

        return """/* -- Dark Mode -- */
@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: #0F172A;
    --color-bg-alt: #1E293B;
    --color-text: #F1F5F9;
    --color-text-muted: #94A3B8;
    --color-divider: #334155;
    --color-shadow: rgba(0,0,0,0.3);
  }

  .evo-card { background: var(--color-bg-alt); }
  .evo-nav { background: rgba(15, 23, 42, 0.95); }
  .evo-footer { background: #0B1120; }
  .evo-toc { background: var(--color-bg-alt); }
  .evo-comparison th { background: #1E293B; }

  img { opacity: 0.9; }
}

[data-theme="dark"] {
  --color-bg: #0F172A;
  --color-bg-alt: #1E293B;
  --color-text: #F1F5F9;
  --color-text-muted: #94A3B8;
  --color-divider: #334155;
  --color-shadow: rgba(0,0,0,0.3);
}
"""

    def generate_accessibility_css(self, ds: DesignSystem) -> str:
        """Generate accessibility-focused styles."""
        return """/* -- Accessibility -- */
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

a:focus-visible, button:focus-visible, input:focus-visible,
select:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

/* Skip to content link */
.skip-to-content {
  position: absolute;
  top: -100%;
  left: var(--space-md);
  padding: var(--space-sm) var(--space-lg);
  background: var(--color-primary);
  color: #fff;
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  z-index: 9999;
  transition: top var(--transition-fast);
}
.skip-to-content:focus { top: 0; }

/* Respect reduced motion preference */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
  [data-animate] { opacity: 1; transform: none; }
}

/* High contrast mode adjustments */
@media (prefers-contrast: high) {
  .evo-btn { border: 2px solid currentColor; }
  .evo-card { border: 1px solid var(--color-text); }
}
"""

    def generate_print_css(self, ds: DesignSystem) -> str:
        """Generate print-friendly styles."""
        return """/* -- Print -- */
@media print {
  .evo-nav, .evo-hero, .evo-newsletter, .evo-footer,
  .evo-cta, .evo-sidebar, .evo-toc,
  .evo-nav__hamburger, .evo-btn { display: none !important; }

  body {
    font-size: 12pt;
    color: #000;
    background: #fff;
    line-height: 1.5;
  }

  a { color: #000; text-decoration: underline; }
  a[href]::after { content: " (" attr(href) ")"; font-size: 0.8em; }

  .evo-card { box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; }

  h1, h2, h3 { page-break-after: avoid; }

  img { max-width: 100% !important; }
}
"""

    def generate_modern_layout_css(self, ds: DesignSystem) -> str:
        """Generate CSS Grid layouts and container queries."""
        return """/* -- Modern Layout -- */
.evo-container { width: 100%; max-width: var(--max-width); margin: 0 auto; padding: 0 var(--space-xl); }

/* CSS Grid named areas */
.evo-layout-sidebar { display: grid; grid-template-columns: 1fr 300px; gap: var(--space-2xl); }
.evo-layout-sidebar__main { min-width: 0; }
.evo-layout-sidebar__side { min-width: 0; }
@media (max-width: 1024px) { .evo-layout-sidebar { grid-template-columns: 1fr; } }

/* Auto-fill responsive grid */
.evo-auto-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--grid-min, 280px), 1fr)); gap: var(--space-xl); }

/* Masonry-like layout via columns */
.evo-masonry { columns: 3 280px; column-gap: var(--space-xl); }
.evo-masonry > * { break-inside: avoid; margin-bottom: var(--space-xl); }

/* Container queries (modern browsers) */
@container (min-width: 600px) { .evo-card--responsive { display: grid; grid-template-columns: 200px 1fr; } }
@container (max-width: 599px) { .evo-card--responsive { display: block; } }
"""

    def generate_scroll_animations_css(self, ds: DesignSystem) -> str:
        """Generate scroll-driven animations and parallax effects."""
        return """/* -- Scroll Animations -- */
@keyframes evo-fadeSlideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
@keyframes evo-fadeSlideLeft { from { opacity: 0; transform: translateX(30px); } to { opacity: 1; transform: translateX(0); } }
@keyframes evo-fadeSlideRight { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
@keyframes evo-zoomIn { from { opacity: 0; transform: scale(0.8); } to { opacity: 1; transform: scale(1); } }

[data-animate="fade-up"] { opacity: 0; transform: translateY(30px); }
[data-animate="fade-up"].visible { animation: evo-fadeSlideUp 0.6s ease forwards; }
[data-animate="fade-left"] { opacity: 0; transform: translateX(30px); }
[data-animate="fade-left"].visible { animation: evo-fadeSlideLeft 0.6s ease forwards; }
[data-animate="fade-right"] { opacity: 0; transform: translateX(-30px); }
[data-animate="fade-right"].visible { animation: evo-fadeSlideRight 0.6s ease forwards; }
[data-animate="zoom"] { opacity: 0; transform: scale(0.8); }
[data-animate="zoom"].visible { animation: evo-zoomIn 0.5s ease forwards; }

/* Staggered children */
[data-animate-stagger] > *:nth-child(1) { animation-delay: 0s; }
[data-animate-stagger] > *:nth-child(2) { animation-delay: 0.1s; }
[data-animate-stagger] > *:nth-child(3) { animation-delay: 0.2s; }
[data-animate-stagger] > *:nth-child(4) { animation-delay: 0.3s; }

/* Parallax helper (requires JS to update transform) */
.evo-parallax { will-change: transform; }
"""

    def generate_micro_interactions_css(self, ds: DesignSystem) -> str:
        """Generate micro-interaction animations for buttons, cards, links."""
        return """/* -- Micro Interactions -- */

/* Button press effect */
.evo-btn:active { transform: scale(0.97); }
.evo-btn-primary:hover { box-shadow: 0 4px 14px rgba(var(--color-primary-rgb, 107,70,193), 0.3); }

/* Card lift on hover */
.evo-card { transition: transform 0.25s ease, box-shadow 0.25s ease; }
.evo-card:hover { transform: translateY(-6px); box-shadow: 0 12px 24px rgba(0,0,0,0.08); }

/* Link underline animation */
.entry-content a, .post-content a {
  text-decoration: none;
  background-image: linear-gradient(var(--color-primary), var(--color-primary));
  background-size: 0% 2px;
  background-position: left bottom;
  background-repeat: no-repeat;
  transition: background-size 0.3s ease;
}
.entry-content a:hover, .post-content a:hover { background-size: 100% 2px; }

/* Input focus glow */
input:focus, textarea:focus, select:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(var(--color-primary-rgb, 107,70,193), 0.15);
}

/* Image hover zoom */
.evo-card__image { transition: transform 0.4s ease; }
.evo-card:hover .evo-card__image { transform: scale(1.05); }

/* Badge pulse on new */
.badge--new { animation: evo-badgePulse 2s ease infinite; }
@keyframes evo-badgePulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(var(--color-primary-rgb, 107,70,193), 0.4); } 50% { box-shadow: 0 0 0 6px transparent; } }
"""

    def generate_dark_mode_toggle_css(self, ds: DesignSystem) -> str:
        """Generate dark mode CSS with [data-theme='dark'] custom property overrides."""
        return """/* -- Dark Mode Toggle Support -- */
[data-theme="dark"] {
  --color-bg: #0F172A;
  --color-bg-alt: #1E293B;
  --color-text: #F1F5F9;
  --color-text-muted: #94A3B8;
  --color-divider: #334155;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 8px rgba(0,0,0,0.3);
  --shadow-lg: 0 8px 24px rgba(0,0,0,0.4);
}

[data-theme="dark"] .evo-card { background: var(--color-bg-alt); border-color: var(--color-divider); }
[data-theme="dark"] .evo-nav { background: rgba(15, 23, 42, 0.95); }
[data-theme="dark"] .evo-footer { background: #0B1120; }
[data-theme="dark"] .evo-toc { background: var(--color-bg-alt); }
[data-theme="dark"] .evo-comparison th { background: #1E293B; }
[data-theme="dark"] .evo-cookie-consent { background: #1E293B; }
[data-theme="dark"] .evo-search__results { background: #1E293B; }
[data-theme="dark"] img { opacity: 0.92; }
[data-theme="dark"] .evo-popup__content { background: #1E293B; color: #F1F5F9; }
"""

    def generate_full_stylesheet(self, ds: DesignSystem) -> str:
        """Generate the complete CSS framework (1000-1800 lines)."""
        sections = [
            f"/* Site Evolution CSS Framework v2.0 — {ds.site_slug} */",
            f"/* Style Lane: {ds.style_lane} */",
            f"/* Generated: {__import__('datetime').datetime.now().isoformat()} */\n",
            self.generate_root_variables(ds),
            self.generate_typography_css(ds),
            self.generate_component_css(ds),
            self.generate_modern_layout_css(ds),
            self.generate_utility_classes(ds),
            self.generate_animation_css(ds),
            self.generate_scroll_animations_css(ds),
            self.generate_micro_interactions_css(ds),
            self.generate_responsive_css(ds),
            self.generate_accessibility_css(ds),
            self.generate_print_css(ds),
            self.generate_dark_mode_css(ds),
            self.generate_dark_mode_toggle_css(ds),
        ]
        full = "\n\n".join(s for s in sections if s)
        log.info("Generated %d-line CSS for %s", full.count("\n"), ds.site_slug)
        return full
