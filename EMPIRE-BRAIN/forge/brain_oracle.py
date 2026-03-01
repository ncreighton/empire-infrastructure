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

    # Impact multipliers for smarter priority scoring
    IMPACT_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 5}
    EFFORT_WEIGHT = {"low": 3, "medium": 2, "high": 1}
    TYPE_BONUS = {
        "monetization": 1.5,    # Revenue opportunities score higher
        "cross_pollination": 1.3,  # Reuse existing work
        "shared_service": 1.2,  # Reduce maintenance
        "architecture": 1.0,
        "monitoring": 0.9,
        "content_gap": 0.8,
        "automation": 0.8,
        "optimization": 1.1,
    }

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def _score_opportunity(self, opp: dict) -> float:
        """Calculate a smarter priority score (0-10) using weighted factors."""
        impact = self.IMPACT_WEIGHT.get(opp.get("impact", "medium"), 2)
        effort = self.EFFORT_WEIGHT.get(opp.get("effort", "medium"), 2)
        type_bonus = self.TYPE_BONUS.get(opp.get("type", "optimization"), 1.0)

        # Base score: impact * effort_ease (higher effort_ease = easier to do)
        base = impact * effort  # 1-15 range

        # Project count bonus: more projects affected = higher value
        affected = opp.get("affected_projects", [])
        project_bonus = min(len(affected) * 0.3, 2.0)  # up to +2

        # Revenue potential: monetization/commerce projects get extra weight
        revenue_projects = {"bmc-witchcraft", "etsy-agent-v2", "witchcraftforbeginners"}
        revenue_overlap = len(set(affected) & revenue_projects)
        revenue_bonus = revenue_overlap * 0.5  # up to +1.5

        raw_score = (base * type_bonus) + project_bonus + revenue_bonus

        # Normalize to 0-10 scale (max raw is ~5*3*1.5+2+1.5 = 26)
        return round(min(10.0, raw_score / 2.6), 1)

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
        """Discover monetization, growth, and architecture opportunities."""
        opportunities = []
        projects = self.db.get_projects()
        conn = self.db._conn()

        # Build lookup structures once
        categories = {}
        slugs_by_cat = {}
        for p in projects:
            cat = p.get("category", "uncategorized")
            categories.setdefault(cat, []).append(p)
            slugs_by_cat.setdefault(cat, set()).add(p["slug"])

        sites = [p for p in projects if "sites" in p.get("category", "")]
        api_projects = [p for p in projects if p.get("endpoint_count", 0) > 0]

        # --- Finder 1: FORGE+AMPLIFY expansion (existing) ---
        opportunities.extend(self._find_forge_expansion(projects, sites))

        # --- Finder 2: Cross-pollination ---
        opportunities.extend(self._find_cross_pollination(projects, conn))

        # --- Finder 3: Shared auth layer ---
        opportunities.extend(self._find_shared_auth(conn))

        # --- Finder 4: API gateway consolidation ---
        opportunities.extend(self._find_api_gateway(api_projects))

        # --- Finder 5: Content pipeline unification ---
        opportunities.extend(self._find_content_unification(conn))

        # --- Finder 6: Revenue attribution ---
        opportunities.extend(self._find_revenue_attribution(conn))

        # --- Finder 7: Monitoring consolidation ---
        opportunities.extend(self._find_monitoring_gaps(api_projects))

        # --- Existing: Video + PinFlux for sites ---
        opportunities.extend(self._find_video_gaps(projects, sites))
        opportunities.extend(self._find_pinflux_gaps(projects, sites))

        conn.close()

        # Score and persist all new opportunities (dedup handled by add_opportunity)
        for opp in opportunities:
            opp["priority_score"] = self._score_opportunity(opp)
            self.db.add_opportunity(
                title=opp["title"],
                opp_type=opp["type"],
                description=opp["description"],
                projects=opp["affected_projects"],
                impact=opp["impact"],
                effort=opp["effort"],
                priority_score=opp["priority_score"],
            )

        # Return sorted by priority score descending
        return sorted(opportunities, key=lambda o: o.get("priority_score", 0), reverse=True)

    # ------------------------------------------------------------------
    # Individual opportunity finders
    # ------------------------------------------------------------------

    def _find_forge_expansion(self, projects: list[dict], sites: list[dict]) -> list[dict]:
        """Find site projects that lack FORGE+AMPLIFY."""
        forge_projects = self.db.get_patterns(pattern_type="architecture")
        forge_slugs = set()
        for p in forge_projects:
            if "forge" in p.get("name", "").lower():
                try:
                    forge_slugs.update(json.loads(p.get("used_by_projects", "[]")))
                except (json.JSONDecodeError, TypeError):
                    pass

        opps = []
        for site_proj in sites:
            if site_proj["slug"] not in forge_slugs:
                opps.append({
                    "title": f"Add FORGE+AMPLIFY to {site_proj['name']}",
                    "type": "optimization",
                    "impact": "high",
                    "effort": "medium",
                    "description": f"Project '{site_proj['name']}' doesn't use FORGE+AMPLIFY pipeline. "
                                   f"Adding it could improve content quality and automation.",
                    "affected_projects": [site_proj["slug"]],
                })
        return opps

    def _find_cross_pollination(self, projects: list[dict], conn) -> list[dict]:
        """Find capabilities in one system that could benefit another."""
        opps = []

        # VideoForge ScriptEngine → ZimmWriter content generation
        vf_scripts = conn.execute(
            "SELECT 1 FROM classes WHERE project_slug = 'videoforge-engine' AND name = 'ScriptEngine'"
        ).fetchone()
        zw_exists = any(p["slug"] == "zimmwriter-project-new" for p in projects)
        if vf_scripts and zw_exists:
            opps.append({
                "title": "Port VideoForge ScriptEngine anti-slop pipeline to ZimmWriter",
                "type": "cross_pollination",
                "impact": "high",
                "effort": "medium",
                "description": "VideoForge's ScriptEngine has an anti-slop pipeline that removes AI-sounding "
                               "language. This same pipeline could clean up ZimmWriter-generated articles, "
                               "improving E-E-A-T across all 16 sites.",
                "affected_projects": ["videoforge-engine", "zimmwriter-project-new"],
            })

        # Grimoire knowledge base → Witchcraft sites content enrichment
        grimoire_knowledge = conn.execute(
            "SELECT COUNT(*) as cnt FROM classes WHERE project_slug = 'grimoire-intelligence' "
            "AND (name LIKE '%Knowledge%' OR name LIKE '%Herb%' OR name LIKE '%Crystal%' OR name LIKE '%Tarot%')"
        ).fetchone()
        if grimoire_knowledge and grimoire_knowledge["cnt"] >= 3:
            opps.append({
                "title": "Use Grimoire knowledge base to enrich witchcraft article content",
                "type": "cross_pollination",
                "impact": "high",
                "effort": "low",
                "description": "Grimoire has 49 herbs, 40 crystals, 78 tarot cards, 8 sabbats cataloged. "
                               "This data could auto-inject accurate correspondences into articles for "
                               "witchcraftforbeginners and manifestandalign, boosting E-E-A-T without AI costs.",
                "affected_projects": ["grimoire-intelligence", "witchcraftforbeginners", "manifestandalign"],
            })

        # Article audit visual checks → Empire dashboard monitoring
        audit_vision = conn.execute(
            "SELECT 1 FROM functions WHERE project_slug = 'article-audit-system' AND name LIKE '%visual%'"
        ).fetchone()
        dash_exists = any(p["slug"] == "empire-dashboard" for p in projects)
        if audit_vision and dash_exists:
            opps.append({
                "title": "Feed article-audit visual regression results into Empire Dashboard",
                "type": "cross_pollination",
                "impact": "medium",
                "effort": "low",
                "description": "Article audit system captures screenshots and detects visual regressions. "
                               "Piping these results as dashboard alerts would catch broken layouts across "
                               "all 16 sites automatically.",
                "affected_projects": ["article-audit-system", "empire-dashboard"],
            })

        # VideoCodex learnings → Brain learnings
        codex_exists = conn.execute(
            "SELECT 1 FROM classes WHERE project_slug = 'videoforge-engine' AND name = 'VideoCodex'"
        ).fetchone()
        if codex_exists:
            opps.append({
                "title": "Sync VideoCodex cost/performance data into EMPIRE-BRAIN learnings",
                "type": "cross_pollination",
                "impact": "medium",
                "effort": "low",
                "description": "VideoCodex tracks per-video costs, render success rates, and niche performance. "
                               "Syncing this into EMPIRE-BRAIN's learnings table would let the Morning Briefing "
                               "surface video ROI trends.",
                "affected_projects": ["videoforge-engine", "empire-brain"],
            })

        return opps

    def _find_shared_auth(self, conn) -> list[dict]:
        """Detect duplicated WordPress auth across projects."""
        opps = []

        # Count projects with their own login_wordpress implementation
        wp_login_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM functions WHERE name = 'login_wordpress'"
        ).fetchall()

        if len(wp_login_projects) >= 8:
            slugs = [r["project_slug"] for r in wp_login_projects]
            opps.append({
                "title": f"Extract shared WordPress auth service ({len(slugs)} duplicate implementations)",
                "type": "shared_service",
                "impact": "high",
                "effort": "medium",
                "description": f"login_wordpress() is copy-pasted across {len(slugs)} projects. "
                               f"A single wp-auth-service or shared module would centralize credential "
                               f"management, simplify app-password rotation, and reduce code by ~{len(slugs) * 30} lines. "
                               f"Projects: {', '.join(slugs[:6])}{'...' if len(slugs) > 6 else ''}",
                "affected_projects": slugs[:10],
            })

        # Also check for duplicated create_post / clear_cache
        for fn_name, label in [
            ("create_post", "WordPress post creation"),
            ("clear_cache", "LiteSpeed cache clearing"),
            ("take_screenshot", "screenshot capture"),
        ]:
            fn_projects = conn.execute(
                "SELECT DISTINCT project_slug FROM functions WHERE name = ?", (fn_name,)
            ).fetchall()
            if len(fn_projects) >= 10:
                slugs = [r["project_slug"] for r in fn_projects]
                opps.append({
                    "title": f"Extract shared {label} utility ({len(slugs)} implementations)",
                    "type": "shared_service",
                    "impact": "medium",
                    "effort": "low",
                    "description": f"{fn_name}() is duplicated across {len(slugs)} projects. "
                                   f"Moving to a shared empire-utils package eliminates maintenance overhead.",
                    "affected_projects": slugs[:8],
                })

        return opps

    def _find_api_gateway(self, api_projects: list[dict]) -> list[dict]:
        """Detect need for API gateway when multiple services run independently."""
        opps = []

        if len(api_projects) >= 5:
            slugs = [p["slug"] for p in api_projects if p.get("endpoint_count", 0) >= 4]
            if len(slugs) >= 4:
                opps.append({
                    "title": f"Add API gateway for {len(slugs)} independent FastAPI services",
                    "type": "architecture",
                    "impact": "high",
                    "effort": "high",
                    "description": f"{len(slugs)} FastAPI services run on separate ports with no unified "
                                   f"routing, auth, or rate limiting. A lightweight gateway (nginx/Traefik) "
                                   f"would add: single entrypoint, shared API key auth, request logging, "
                                   f"health aggregation, and CORS in one place. "
                                   f"Services: {', '.join(slugs[:6])}",
                    "affected_projects": slugs[:8],
                })

        # Check for services without /health endpoint
        no_health = []
        conn = self.db._conn()
        for p in api_projects:
            if p.get("endpoint_count", 0) >= 4:
                has_health = conn.execute(
                    "SELECT 1 FROM api_endpoints WHERE project_slug = ? AND path LIKE '%health%'",
                    (p["slug"],)
                ).fetchone()
                if not has_health:
                    no_health.append(p["slug"])
        conn.close()

        if no_health:
            opps.append({
                "title": f"Add /health endpoint to {len(no_health)} API services",
                "type": "architecture",
                "impact": "medium",
                "effort": "low",
                "description": f"These API services lack a /health endpoint, making automated monitoring "
                               f"impossible: {', '.join(no_health)}",
                "affected_projects": no_health,
            })

        return opps

    def _find_content_unification(self, conn) -> list[dict]:
        """Detect fragmented content generation across multiple systems."""
        opps = []

        # Find all projects that generate articles
        content_projects = conn.execute("""
            SELECT DISTINCT project_slug FROM functions
            WHERE name IN ('generate_article', 'write_article', 'generate_content',
                           '_generate_article', 'create_article')
            AND project_slug NOT LIKE '%test%'
            AND project_slug != '_archive'
        """).fetchall()

        slugs = [r["project_slug"] for r in content_projects]
        if len(slugs) >= 3:
            opps.append({
                "title": f"Unify content generation across {len(slugs)} separate pipelines",
                "type": "architecture",
                "impact": "critical",
                "effort": "high",
                "description": f"Article generation is fragmented across {len(slugs)} projects: "
                               f"{', '.join(slugs)}. Each has its own prompt engineering, model selection, "
                               f"and quality checks. A single content-engine service would enforce consistent "
                               f"brand voice, E-E-A-T signals, anti-slop filtering, and cost-optimized "
                               f"model routing across all 16 sites.",
                "affected_projects": slugs,
            })

        # Check for projects using different AI models/APIs
        openrouter_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM functions WHERE name LIKE '%openrouter%'"
        ).fetchall()
        claude_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM functions WHERE name LIKE '%claude%' OR name LIKE '%anthropic%'"
        ).fetchall()

        or_slugs = {r["project_slug"] for r in openrouter_projects}
        cl_slugs = {r["project_slug"] for r in claude_projects}
        mixed = or_slugs & cl_slugs
        if mixed:
            opps.append({
                "title": "Standardize LLM provider routing across projects",
                "type": "optimization",
                "impact": "medium",
                "effort": "medium",
                "description": f"{len(mixed)} projects use both OpenRouter and direct Claude API. "
                               f"A shared LLM router would optimize cost (route simple tasks to Haiku, "
                               f"complex to Sonnet) and add prompt caching uniformly. "
                               f"Projects: {', '.join(list(mixed)[:5])}",
                "affected_projects": list(mixed)[:8],
            })

        return opps

    def _find_revenue_attribution(self, conn) -> list[dict]:
        """Detect gaps in revenue tracking and attribution."""
        opps = []

        # Check if BMC webhook handler exists but lacks content attribution
        bmc_exists = conn.execute(
            "SELECT 1 FROM functions WHERE project_slug = 'bmc-witchcraft' AND name = 'handle_bmc_webhook'"
        ).fetchone()
        # Check if any revenue tracking connects to content
        revenue_to_content = conn.execute("""
            SELECT 1 FROM functions
            WHERE name LIKE '%revenue%content%' OR name LIKE '%attribution%'
            OR name LIKE '%revenue%article%' OR name LIKE '%conversion%track%'
        """).fetchone()

        if bmc_exists and not revenue_to_content:
            opps.append({
                "title": "Add revenue-to-content attribution for BMC + AdSense",
                "type": "monetization",
                "impact": "high",
                "effort": "medium",
                "description": "BMC webhook captures payment events but nothing traces which articles, "
                               "videos, or email sequences drove each sale. Adding UTM tracking from "
                               "content → BMC page → webhook would reveal which content converts, "
                               "informing the content calendar. Same pattern applies to AdSense per-page revenue.",
                "affected_projects": ["bmc-witchcraft", "empire-dashboard", "witchcraftforbeginners"],
            })

        # Check for AdSense/monetization in site projects
        adsense_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM functions WHERE name LIKE '%adsense%' OR name LIKE '%monetiz%'"
        ).fetchall()
        site_slugs = {p["project_slug"] for p in adsense_projects}

        # Check which EMPIRE_SITES have no monetization code at all
        unmonetized = []
        for site in EMPIRE_SITES:
            has_any_revenue = conn.execute("""
                SELECT 1 FROM functions WHERE project_slug = ?
                AND (name LIKE '%revenue%' OR name LIKE '%adsense%' OR name LIKE '%affiliate%'
                     OR name LIKE '%monetiz%' OR name LIKE '%shop%')
            """, (site,)).fetchone()
            if not has_any_revenue:
                unmonetized.append(site)

        if len(unmonetized) >= 5:
            opps.append({
                "title": f"Add monetization tracking to {len(unmonetized)} sites with no revenue code",
                "type": "monetization",
                "impact": "high",
                "effort": "medium",
                "description": f"These sites have zero monetization-related functions: "
                               f"{', '.join(unmonetized[:8])}. At minimum, each should track "
                               f"AdSense RPM per article and affiliate click-through rates.",
                "affected_projects": unmonetized[:10],
            })

        return opps

    def _find_monitoring_gaps(self, api_projects: list[dict]) -> list[dict]:
        """Detect services not covered by the dashboard health monitor."""
        opps = []

        # Known monitored services (from settings.py SERVICES dict)
        monitored_ports = {3030, 8000, 8002, 8080, 8090, 8095, 8200, 8765}
        monitored_slugs = {
            "screenpipe", "empire-dashboard", "geelark-automation",
            "grimoire-intelligence", "videoforge-engine", "bmc-witchcraft",
            "empire-brain", "zimmwriter-project-new"
        }

        unmonitored = []
        for p in api_projects:
            if p.get("endpoint_count", 0) >= 4 and p["slug"] not in monitored_slugs:
                # Skip infrastructure/archive projects
                if p.get("category") in ("infrastructure",) and p["slug"] in (
                    "_archive", "scripts", "src", "project-mesh-v2-omega"
                ):
                    continue
                unmonitored.append(p["slug"])

        if unmonitored:
            opps.append({
                "title": f"Add {len(unmonitored)} API services to dashboard health monitoring",
                "type": "monitoring",
                "impact": "medium",
                "effort": "low",
                "description": f"These API services have endpoints but aren't in the dashboard health "
                               f"monitor: {', '.join(unmonitored)}. Adding them to SERVICES in "
                               f"config/settings.py and the dashboard health card catches outages early.",
                "affected_projects": unmonitored + ["empire-dashboard"],
            })

        # Check for projects with high function count but no logging
        conn = self.db._conn()
        no_logging = []
        for p in api_projects:
            if p.get("function_count", 0) >= 50:
                has_logging = conn.execute(
                    "SELECT 1 FROM functions WHERE project_slug = ? "
                    "AND (name LIKE '%log%' OR name LIKE '%logger%') LIMIT 1",
                    (p["slug"],)
                ).fetchone()
                if not has_logging:
                    no_logging.append(p["slug"])
        conn.close()

        if no_logging:
            opps.append({
                "title": f"Add structured logging to {len(no_logging)} services",
                "type": "monitoring",
                "impact": "medium",
                "effort": "low",
                "description": f"These services have no logging functions, making debugging impossible: "
                               f"{', '.join(no_logging)}",
                "affected_projects": no_logging,
            })

        return opps

    def _find_video_gaps(self, projects: list[dict], sites: list[dict]) -> list[dict]:
        """Find sites that could use VideoForge but don't."""
        opps = []
        video_systems = [p for p in projects if p.get("category") == "video-systems"]
        if video_systems:
            for site in sites:
                site_name = site.get("name", "").lower()
                if any(v in site_name for v in ["witchcraft", "smart"]):
                    continue
                opps.append({
                    "title": f"Create video content for {site['name']}",
                    "type": "content_gap",
                    "impact": "medium",
                    "effort": "low",
                    "description": f"VideoForge engine exists but {site['name']} has no video pipeline. "
                                   f"Could drive traffic via YouTube/TikTok.",
                    "affected_projects": [site["slug"], "videoforge-engine"],
                })
        return opps

    def _find_pinflux_gaps(self, projects: list[dict], sites: list[dict]) -> list[dict]:
        """Find sites that could use PinFlux but don't."""
        opps = []
        pinflux = [p for p in projects if "pinflux" in p.get("slug", "")]
        if pinflux:
            for site in sites:
                opps.append({
                    "title": f"Enable PinFlux for {site['name']}",
                    "type": "automation",
                    "impact": "medium",
                    "effort": "low",
                    "description": f"PinFlux engine can auto-generate pins from {site['name']} blog posts.",
                    "affected_projects": [site["slug"], "pinflux-engine"],
                })
        return opps

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
        """Generate prioritized action recommendations with actionable next steps."""
        recs = []

        # Action templates by opportunity type
        action_map = {
            "monetization": "Set up tracking → measure baseline → A/B test → scale",
            "cross_pollination": "Import module → create adapter → wire into pipeline → test",
            "shared_service": "Extract to shared/ → update imports in consumers → add tests",
            "architecture": "Design interface → build prototype → migrate one consumer → rollout",
            "monitoring": "Add /health endpoint → register in dashboard → set alert thresholds",
            "content_gap": "Create content brief → generate draft → review → publish",
            "automation": "Configure trigger → build workflow → test with sample → enable",
            "optimization": "Profile current → identify bottleneck → apply fix → benchmark",
        }

        opportunities = self.db.get_opportunities(status="open")
        for opp in opportunities[:10]:
            opp_type = opp.get("opportunity_type", "general")
            try:
                projects = json.loads(opp.get("affected_projects", "[]"))
            except (json.JSONDecodeError, TypeError):
                projects = []

            recs.append({
                "title": opp["title"],
                "priority": opp.get("priority_score", 0),
                "type": opp_type,
                "impact": opp.get("estimated_impact", "medium"),
                "effort": opp.get("estimated_effort", "medium"),
                "affected_projects": projects[:5],
                "next_steps": action_map.get(opp_type, "Analyze → plan → implement → verify"),
            })

        return sorted(recs, key=lambda r: r["priority"], reverse=True)
