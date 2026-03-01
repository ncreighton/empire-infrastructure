"""BrainOracle — Predictive Intelligence Engine

Predicts trends, risks, and opportunities across the empire:
- Project growth trajectories
- Content gap analysis
- Monetization opportunities
- Risk forecasting
- Resource optimization suggestions
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB
from config.settings import EMPIRE_SITES


class BrainOracle:
    """Predicts opportunities and risks across the empire."""

    # Niche growth potential (based on market analysis)
    NICHE_POTENTIAL = {
        "witchcraft-sites": {"growth": "high", "competition": "medium", "monetization": "high"},
        "ai-sites": {"growth": "very-high", "competition": "high", "monetization": "high"},
        "tech-sites": {"growth": "medium", "competition": "high", "monetization": "medium"},
        "lifestyle-sites": {"growth": "medium", "competition": "medium", "monetization": "medium"},
        "commerce": {"growth": "high", "competition": "medium", "monetization": "very-high"},
    }

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def weekly_forecast(self) -> dict:
        """Generate comprehensive weekly forecast."""
        return {
            "timestamp": datetime.now().isoformat(),
            "opportunities": self.find_opportunities(),
            "risks": self.assess_risks(),
            "recommendations": self.generate_recommendations(),
            "content_gaps": self.find_content_gaps(),
            "optimization_targets": self.find_optimizations(),
        }

    def find_opportunities(self) -> list[dict]:
        """Discover monetization and growth opportunities."""
        opportunities = []
        projects = self.db.get_projects()

        # Group by category
        categories = {}
        for p in projects:
            cat = p.get("category", "uncategorized")
            categories.setdefault(cat, []).append(p)

        # Cross-pollination opportunities
        systems = [p for p in projects if p.get("category") in ("video-systems", "infrastructure", "content-tools")]
        sites = [p for p in projects if "sites" in p.get("category", "")]

        # FORGE+AMPLIFY expansion
        forge_projects = self.db.get_patterns(pattern_type="architecture")
        forge_slugs = set()
        for p in forge_projects:
            if "forge" in p.get("name", "").lower():
                try:
                    forge_slugs.update(json.loads(p.get("used_by_projects", "[]")))
                except (json.JSONDecodeError, TypeError):
                    pass

        for site_proj in sites:
            if site_proj["slug"] not in forge_slugs:
                opp = {
                    "title": f"Add FORGE+AMPLIFY to {site_proj['name']}",
                    "type": "optimization",
                    "impact": "high",
                    "effort": "medium",
                    "description": f"Project '{site_proj['name']}' doesn't use FORGE+AMPLIFY pipeline. Adding it could improve content quality and automation.",
                    "affected_projects": [site_proj["slug"]],
                }
                opportunities.append(opp)
                self.db.add_opportunity(**{
                    "title": opp["title"],
                    "opp_type": opp["type"],
                    "description": opp["description"],
                    "projects": opp["affected_projects"],
                    "impact": opp["impact"],
                    "effort": opp["effort"],
                })

        # Video creation for sites without video
        video_systems = [p for p in projects if p.get("category") == "video-systems"]
        if video_systems:
            for site in sites:
                site_name = site.get("name", "").lower()
                # These already have video
                if any(v in site_name for v in ["witchcraft", "smart"]):
                    continue
                opp = {
                    "title": f"Create video content for {site['name']}",
                    "type": "content_gap",
                    "impact": "medium",
                    "effort": "low",
                    "description": f"VideoForge engine exists but {site['name']} has no video content pipeline. Could drive traffic via YouTube/TikTok.",
                    "affected_projects": [site["slug"], "videoforge-engine"],
                }
                opportunities.append(opp)

        # Pinterest automation for sites
        pinflux = [p for p in projects if "pinflux" in p.get("slug", "")]
        if pinflux:
            for site in sites:
                opp = {
                    "title": f"Enable PinFlux for {site['name']}",
                    "type": "automation",
                    "impact": "medium",
                    "effort": "low",
                    "description": f"PinFlux engine can auto-generate pins from {site['name']} blog posts.",
                    "affected_projects": [site["slug"], "pinflux-engine"],
                }
                opportunities.append(opp)

        return opportunities[:20]

    def assess_risks(self) -> list[dict]:
        """Identify current risks across the empire."""
        risks = []
        projects = self.db.get_projects()

        for proj in projects:
            score = proj.get("health_score", 0)
            if score < 40:
                risks.append({
                    "project": proj["slug"],
                    "risk": "health_critical",
                    "severity": "high",
                    "detail": f"Health score {score}/100 — needs immediate attention",
                })
            elif score < 60:
                risks.append({
                    "project": proj["slug"],
                    "risk": "health_degraded",
                    "severity": "medium",
                    "detail": f"Health score {score}/100 — trending down",
                })

        # Check for projects with zero tests
        for proj in projects:
            path = Path(proj.get("path", ""))
            if path.exists() and not any(path.rglob("test_*.py")):
                if proj.get("function_count", 0) > 10:
                    risks.append({
                        "project": proj["slug"],
                        "risk": "no_tests",
                        "severity": "medium",
                        "detail": f"{proj.get('function_count', 0)} functions but no tests",
                    })

        return sorted(risks, key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r["severity"], 3))

    def find_content_gaps(self) -> list[dict]:
        """Identify content gaps across sites."""
        gaps = []
        # Cross-reference site skills to find which sites lack capabilities others have
        skills = self.db.get_skills()
        skill_by_project = {}
        for s in skills:
            proj = s.get("project_slug", "")
            skill_by_project.setdefault(proj, set()).add(s.get("category", ""))

        all_categories = set()
        for cats in skill_by_project.values():
            all_categories.update(cats)

        for proj, cats in skill_by_project.items():
            missing = all_categories - cats
            if missing:
                gaps.append({
                    "project": proj,
                    "missing_capabilities": list(missing),
                    "suggestion": f"Could add: {', '.join(list(missing)[:3])}",
                })

        return gaps

    def find_optimizations(self) -> list[dict]:
        """Find efficiency improvements."""
        opts = []

        # Find duplicate patterns that should be extracted to shared-core
        patterns = self.db.get_patterns(pattern_type="code_pattern")
        for p in patterns:
            if p.get("frequency", 0) >= 3:
                try:
                    projects = json.loads(p.get("used_by_projects", "[]"))
                except (json.JSONDecodeError, TypeError):
                    projects = []
                opts.append({
                    "type": "extract_to_shared",
                    "pattern": p["name"],
                    "frequency": p["frequency"],
                    "projects": projects,
                    "suggestion": f"Extract '{p['name']}' to shared-core — used in {p['frequency']} projects",
                })

        return opts

    def generate_recommendations(self) -> list[dict]:
        """Generate prioritized action recommendations."""
        recs = []

        opportunities = self.db.get_opportunities(status="open")
        for opp in opportunities[:10]:
            recs.append({
                "title": opp["title"],
                "priority": opp.get("priority_score", 0),
                "type": opp.get("opportunity_type", "general"),
                "impact": opp.get("estimated_impact", "medium"),
                "effort": opp.get("estimated_effort", "medium"),
            })

        return sorted(recs, key=lambda r: r["priority"], reverse=True)
