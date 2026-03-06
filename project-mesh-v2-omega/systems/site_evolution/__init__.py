"""
Site Evolution Engine — AI-Driven Design & Enhancement for 14 WordPress Sites.

Transforms every site from basic WordPress to premium, conversion-optimized,
SEO-dominant — and keeps improving them forever.

8 sub-systems:
1. WP Deployer    — Push CSS/PHP/pages/snippets to all 14 sites
2. Design Generator — AI-driven per-site design from brand config + style lanes
3. Component Factory — 12 component types per site
4. SEO Maximizer   — Schema, meta, LLMO, GSC + Bing analytics
5. Performance     — Core Web Vitals, critical CSS, lazy loading
6. Site Auditor    — 8-dimension scoring with 120+ checks
7. Enhancement Queue — Priority-ranked improvements with auto-execution
8. Orchestrator    — AUDIT → PLAN → DESIGN → DEPLOY → VERIFY cycle
"""

from systems.site_evolution.orchestrator import SiteEvolutionEngine

__all__ = ["SiteEvolutionEngine"]
__version__ = "1.0.0"
