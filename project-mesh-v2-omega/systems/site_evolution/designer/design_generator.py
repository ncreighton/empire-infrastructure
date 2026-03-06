"""
Design Generator — Master design system generator.
Reads brand config + style lanes and produces a complete DesignSystem per site.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

CONFIG_ROOT = Path(r"D:\Claude Code Projects\config")

# Site slug → style lane mapping (canonical)
SITE_LANE_MAP = {
    "witchcraftforbeginners": "earthy_herbalist",
    "smarthomewizards": "minimal_clean",
    "mythicalarchives": "dark_lux_mystical",
    "bulletjournals": "premium_editorial",
    "wealthfromai": "premium_editorial",
    "aidiscoverydigest": "minimal_clean",
    "aiinactionhub": "bold_pop_modern",
    "pulsegearreviews": "bold_pop_modern",
    "wearablegearreviews": "minimal_clean",
    "smarthomegearreviews": "minimal_clean",
    "clearainews": "bold_pop_modern",
    "theconnectedhaven": "soft_pastel_cozy",
    "manifestandalign": "celestial_night_sky",
    "familyflourish": "soft_pastel_cozy",
}

# Sites that support dark mode
DARK_MODE_SITES = {"clearainews", "wealthfromai", "aidiscoverydigest", "aiinactionhub"}


@dataclass
class DesignSystem:
    """Complete design system for a site."""
    site_slug: str
    style_lane: str
    css_variables: Dict = field(default_factory=dict)
    component_styles: Dict = field(default_factory=dict)
    typography_stack: Dict = field(default_factory=dict)
    color_palette: Dict = field(default_factory=dict)
    spacing_scale: Dict = field(default_factory=dict)
    animation_presets: Dict = field(default_factory=dict)
    responsive_breakpoints: Dict = field(default_factory=dict)
    supports_dark_mode: bool = False


class DesignGenerator:
    """Generate complete design systems from brand config + style lanes."""

    def __init__(self):
        self._lanes = self._load_lanes()
        self._sites = self._load_sites()

    def _load_lanes(self) -> Dict:
        path = CONFIG_ROOT / "style_lanes.json"
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            return data.get("lanes", {})
        return {}

    def _load_sites(self) -> Dict:
        path = CONFIG_ROOT / "sites.json"
        if path.exists():
            data = json.loads(path.read_text("utf-8"))
            return data.get("sites", data)
        return {}

    def generate_design_system(self, site_slug: str) -> DesignSystem:
        """Generate a complete design system for a site."""
        lane_name = SITE_LANE_MAP.get(site_slug, "minimal_clean")
        lane = self._lanes.get(lane_name, {})
        site = self._sites.get(site_slug, {})
        brand = site.get("brand", {})
        colors = brand.get("colors", {})
        fonts = brand.get("fonts", {})

        palette = lane.get("palette", {})
        typography = lane.get("typography", {})
        shapes = lane.get("shapes", {})
        layout = lane.get("layout", {})

        # Build color palette (merge lane defaults with brand overrides)
        color_palette = {
            "primary": colors.get("primary", palette.get("accent_1", "#3B82F6")),
            "secondary": colors.get("secondary", palette.get("accent_2", "#10B981")),
            "accent": colors.get("accent", palette.get("accent_1", "#3B82F6")),
            "bg_primary": palette.get("bg_primary", "#FFFFFF"),
            "bg_secondary": palette.get("bg_secondary", "#F8F9FA"),
            "text_primary": palette.get("text_primary", "#1A1A1A"),
            "text_secondary": palette.get("text_secondary", "#6B7280"),
            "badge_bg": palette.get("badge_bg", "#1A1A1A"),
            "badge_text": palette.get("badge_text", "#FFFFFF"),
            "divider": palette.get("divider", "#E5E7EB"),
            "shadow": palette.get("shadow", "rgba(0,0,0,0.06)"),
        }

        # Typography stack
        headline_font = fonts.get("headline", typography.get("headline", {}).get("family", "Inter"))
        body_font = fonts.get("body", typography.get("body", {}).get("family", "Inter"))
        typography_stack = {
            "headline": {
                "family": headline_font,
                "weight": typography.get("headline", {}).get("weight", 700),
                "tracking": typography.get("headline", {}).get("tracking", "-0.02em"),
            },
            "subhead": {
                "family": typography.get("subhead", {}).get("family", body_font),
                "weight": typography.get("subhead", {}).get("weight", 300),
                "style": typography.get("subhead", {}).get("style", "normal"),
            },
            "body": {
                "family": body_font,
                "weight": typography.get("body", {}).get("weight", 400),
                "tracking": typography.get("body", {}).get("tracking", "0.01em"),
            },
            "accent": typography.get("accent", {"family": headline_font, "weight": 500}),
            "badge": typography.get("badge", {"family": body_font, "weight": 600}),
        }

        # Spacing scale (modular, based on layout density)
        density = layout.get("density", "medium")
        base = 16 if density in ("low", "low_medium") else 14 if density == "medium" else 12
        spacing_scale = {
            "xs": f"{base * 0.25}px",
            "sm": f"{base * 0.5}px",
            "md": f"{base}px",
            "lg": f"{base * 1.5}px",
            "xl": f"{base * 2}px",
            "2xl": f"{base * 3}px",
            "3xl": f"{base * 4}px",
            "section": f"{base * 5}px",
        }

        # CSS variables (50+)
        border_radius = shapes.get("border_radius", "4px")
        if "for circles" in str(border_radius):
            border_radius = "2px"

        css_variables = {
            # Colors
            "--color-primary": color_palette["primary"],
            "--color-secondary": color_palette["secondary"],
            "--color-accent": color_palette["accent"],
            "--color-bg": color_palette["bg_primary"],
            "--color-bg-alt": color_palette["bg_secondary"],
            "--color-text": color_palette["text_primary"],
            "--color-text-muted": color_palette["text_secondary"],
            "--color-badge-bg": color_palette["badge_bg"],
            "--color-badge-text": color_palette["badge_text"],
            "--color-divider": color_palette["divider"],
            "--color-shadow": color_palette["shadow"],
            # Typography
            "--font-headline": f"'{headline_font}', serif",
            "--font-body": f"'{body_font}', sans-serif",
            "--font-weight-headline": str(typography_stack["headline"]["weight"]),
            "--font-weight-body": str(typography_stack["body"]["weight"]),
            "--letter-spacing-headline": typography_stack["headline"]["tracking"],
            "--letter-spacing-body": typography_stack["body"]["tracking"],
            "--line-height-headline": "1.2",
            "--line-height-body": "1.7",
            # Sizing
            "--font-size-xs": "0.75rem",
            "--font-size-sm": "0.875rem",
            "--font-size-base": "1rem",
            "--font-size-lg": "1.125rem",
            "--font-size-xl": "1.25rem",
            "--font-size-2xl": "1.5rem",
            "--font-size-3xl": "1.875rem",
            "--font-size-4xl": "2.25rem",
            "--font-size-5xl": "3rem",
            # Spacing
            "--space-xs": spacing_scale["xs"],
            "--space-sm": spacing_scale["sm"],
            "--space-md": spacing_scale["md"],
            "--space-lg": spacing_scale["lg"],
            "--space-xl": spacing_scale["xl"],
            "--space-2xl": spacing_scale["2xl"],
            "--space-3xl": spacing_scale["3xl"],
            "--space-section": spacing_scale["section"],
            # Shapes
            "--radius-sm": "2px" if border_radius == "0px" else border_radius,
            "--radius-md": border_radius,
            "--radius-lg": str(int(border_radius.replace("px", "") or "4") * 2) + "px" if "px" in border_radius else border_radius,
            "--radius-full": "9999px",
            # Shadows
            "--shadow-sm": f"0 1px 2px {color_palette['shadow']}",
            "--shadow-md": f"0 4px 6px {color_palette['shadow']}",
            "--shadow-lg": f"0 10px 15px {color_palette['shadow']}",
            "--shadow-xl": f"0 20px 25px {color_palette['shadow']}",
            # Layout
            "--max-width": "1200px",
            "--max-width-content": "800px",
            "--max-width-wide": "1400px",
            "--whitespace-ratio": str(layout.get("whitespace_ratio", 0.35)),
            # Transitions
            "--transition-fast": "150ms ease",
            "--transition-base": "250ms ease",
            "--transition-slow": "400ms ease",
        }

        # Animation presets
        animation_presets = {
            "fadeIn": "fadeIn 0.6s ease forwards",
            "slideUp": "slideUp 0.5s ease forwards",
            "scaleIn": "scaleIn 0.4s ease forwards",
            "pulse": "pulse 2s ease-in-out infinite",
            "shimmer": "shimmer 2s linear infinite",
        }

        # Responsive breakpoints
        responsive_breakpoints = {
            "xs": "480px",
            "sm": "640px",
            "md": "768px",
            "lg": "1024px",
            "xl": "1280px",
        }

        # Component styles (per-component overrides)
        component_styles = {
            "hero": {
                "min_height": "70vh" if density in ("low", "low_medium") else "60vh",
                "padding": spacing_scale["section"],
            },
            "cards": {
                "border_radius": border_radius,
                "shadow": f"0 4px 6px {color_palette['shadow']}",
            },
            "nav": {
                "height": "70px",
                "backdrop_blur": "12px",
            },
        }

        ds = DesignSystem(
            site_slug=site_slug,
            style_lane=lane_name,
            css_variables=css_variables,
            component_styles=component_styles,
            typography_stack=typography_stack,
            color_palette=color_palette,
            spacing_scale=spacing_scale,
            animation_presets=animation_presets,
            responsive_breakpoints=responsive_breakpoints,
            supports_dark_mode=site_slug in DARK_MODE_SITES,
        )

        # Persist to codex
        from systems.site_evolution import codex
        codex.save_design_system(
            site_slug, lane_name, css_variables,
            typography_stack, color_palette, component_styles
        )

        log.info("Generated design system for %s (lane=%s, vars=%d)",
                 site_slug, lane_name, len(css_variables))
        return ds

    def generate_all(self) -> Dict[str, DesignSystem]:
        """Generate design systems for all 14 sites."""
        results = {}
        for slug in SITE_LANE_MAP:
            try:
                results[slug] = self.generate_design_system(slug)
            except Exception as e:
                log.error("Failed to generate design for %s: %s", slug, e)
        return results
