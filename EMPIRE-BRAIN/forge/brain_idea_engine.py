"""BrainIdeaEngine — Innovation Generator

Analyzes the empire to find:
- Feature gaps between similar projects
- Enhancement opportunities from high-frequency patterns
- New project ideas from combining existing capabilities
- Cross-pollination opportunities using function signatures + behavior
- Automation opportunities from repeated manual processes

All ideas are stored in the ideas table for human review.
Zero AI API cost — all algorithmic analysis from indexed DB data.

Improvements over v1:
- Transparent scoring matrix (impact × effort_ease × type_bonus × breadth)
- Algorithmic synergy detection — no hardcoded map, discovers combos from capabilities
- Cross-category gap analysis (not just same-category)
- try/finally on all DB connections
- evolution_id tracking on all stored ideas
- Cached project data to avoid repeated queries
- Semantic function matching (verb stems, not just exact names)
"""
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB
from config.settings import EMPIRE_ROOT

log = logging.getLogger("evolution-engine")

# Verbs that indicate valuable capabilities (for function analysis)
CAPABILITY_VERBS = {
    "generate", "create", "build", "produce", "render", "compose",
    "analyze", "scan", "detect", "discover", "audit", "check", "validate",
    "optimize", "enhance", "improve", "upgrade", "refactor",
    "deploy", "upload", "publish", "push", "sync",
    "monitor", "watch", "track", "alert", "notify",
    "convert", "transform", "parse", "extract", "migrate",
    "schedule", "automate", "orchestrate", "pipeline", "batch",
    "export", "import", "backup", "restore",
}

# Stem mapping for semantic matching (group similar function verbs)
VERB_STEMS = {
    "generate": "create", "build": "create", "produce": "create", "render": "create", "compose": "create",
    "scan": "analyze", "detect": "analyze", "discover": "analyze", "audit": "analyze",
    "check": "validate", "verify": "validate", "test": "validate",
    "deploy": "publish", "upload": "publish", "push": "publish",
    "monitor": "watch", "track": "watch", "alert": "notify",
    "convert": "transform", "parse": "transform", "extract": "transform", "migrate": "transform",
    "schedule": "automate", "orchestrate": "automate", "pipeline": "automate", "batch": "automate",
}

# Category compatibility for cross-pollination (higher = more compatible)
CATEGORY_COMPATIBILITY = {
    ("witchcraft-sites", "lifestyle-sites"): 0.7,
    ("tech-sites", "ai-sites"): 0.8,
    ("tech-sites", "tech-reviews"): 0.9,
    ("ai-sites", "tech-reviews"): 0.6,
    ("infrastructure", "automation"): 0.9,
    ("infrastructure", "content-tools"): 0.7,
    ("automation", "content-tools"): 0.8,
    ("commerce", "content-tools"): 0.6,
}


class BrainIdeaEngine:
    """Generates innovation ideas from empire-wide analysis."""

    # Transparent scoring constants
    IMPACT_SCORES = {"low": 1.0, "medium": 2.0, "high": 3.5, "critical": 5.0}
    EFFORT_EASE = {"low": 3.0, "medium": 2.0, "high": 1.0}  # inverted: low effort = high score
    TYPE_BONUSES = {
        "new_project": 1.4,
        "cross_pollination": 1.3,
        "automation": 1.25,
        "feature_gap": 1.1,
        "enhancement": 1.0,
    }
    MAX_SCORE = 10.0

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()
        self._project_cache = None  # Lazy-loaded

    def _get_projects_cached(self) -> list[dict]:
        """Cache project list to avoid repeated DB queries within a single pass."""
        if self._project_cache is None:
            self._project_cache = self.db.get_projects()
        return self._project_cache

    def _get_project_capabilities(self, slug: str, conn) -> dict:
        """Get capabilities for a single project from DB."""
        funcs = conn.execute(
            "SELECT DISTINCT name FROM functions WHERE project_slug = ?", (slug,)
        ).fetchall()
        func_names = {r["name"] for r in funcs}

        eps = conn.execute(
            "SELECT path FROM api_endpoints WHERE project_slug = ?", (slug,)
        ).fetchall()
        ep_paths = {r["path"] for r in eps}

        classes = conn.execute(
            "SELECT name, methods_count FROM classes WHERE project_slug = ?", (slug,)
        ).fetchall()
        class_names = {r["name"] for r in classes}

        # Extract capability verbs from function names
        capability_verbs = set()
        for fn in func_names:
            parts = fn.lower().replace("-", "_").split("_")
            for part in parts:
                if part in CAPABILITY_VERBS:
                    stem = VERB_STEMS.get(part, part)
                    capability_verbs.add(stem)

        return {
            "functions": func_names,
            "endpoints": ep_paths,
            "classes": class_names,
            "capability_verbs": capability_verbs,
            "func_count": len(func_names),
            "has_tests": any("test" in f.lower() for f in func_names),
            "has_health": any("health" in e for e in ep_paths),
            "has_api": len(ep_paths) > 0,
            "has_auth": any(kw in fn.lower() for fn in func_names for kw in ["auth", "login", "token"]),
            "has_monitoring": any(kw in fn.lower() for fn in func_names for kw in ["monitor", "alert", "watch", "health"]),
        }

    def find_feature_gaps(self) -> list[dict]:
        """Compare projects within same category, find capability differences."""
        projects = self._get_projects_cached()
        conn = self.db._conn()
        gaps = []

        try:
            # Group projects by category
            by_category = defaultdict(list)
            for p in projects:
                cat = p.get("category", "uncategorized")
                by_category[cat].append(p)

            for cat, cat_projects in by_category.items():
                if len(cat_projects) < 2:
                    continue

                # Get capabilities for each project
                project_caps = {}
                for p in cat_projects:
                    slug = p["slug"]
                    project_caps[slug] = self._get_project_capabilities(slug, conn)

                # Compare all pairs within category
                slugs = list(project_caps.keys())
                for i, slug_a in enumerate(slugs):
                    for slug_b in slugs[i + 1:]:
                        caps_a = project_caps[slug_a]
                        caps_b = project_caps[slug_b]

                        # API gap: one has API, other doesn't but has enough code
                        if caps_a["has_api"] and not caps_b["has_api"] and caps_b["func_count"] >= 10:
                            gaps.append({
                                "title": f"Add API layer to {slug_b}",
                                "idea_type": "feature_gap",
                                "description": f"{slug_a} has API endpoints but {slug_b} (same category: {cat}) does not, despite having {caps_b['func_count']} functions",
                                "rationale": "API layer enables MCP integration and remote access",
                                "affected_projects": [slug_b, slug_a],
                                "impact": "medium",
                                "effort": "medium",
                            })

                        # Testing gap
                        if caps_a["has_tests"] and not caps_b["has_tests"] and caps_b["func_count"] >= 10:
                            gaps.append({
                                "title": f"Add test suite to {slug_b}",
                                "idea_type": "feature_gap",
                                "description": f"{slug_a} has tests but {slug_b} ({cat} category, {caps_b['func_count']} functions) does not",
                                "rationale": "Tests prevent regressions in critical code",
                                "affected_projects": [slug_b],
                                "impact": "medium",
                                "effort": "medium",
                            })

                        # Monitoring gap
                        if caps_a["has_monitoring"] and not caps_b["has_monitoring"] and caps_b["has_api"]:
                            gaps.append({
                                "title": f"Add monitoring to {slug_b}",
                                "idea_type": "feature_gap",
                                "description": f"{slug_a} has monitoring/alerting but {slug_b} (API service) does not",
                                "rationale": "API services need health monitoring for reliability",
                                "affected_projects": [slug_b],
                                "impact": "medium",
                                "effort": "low",
                            })

                        # Capability verb gaps: valuable verbs in A not in B
                        verb_gap = caps_a["capability_verbs"] - caps_b["capability_verbs"]
                        if len(verb_gap) >= 2 and caps_b["func_count"] >= 5:
                            # Find actual functions for those verbs
                            valuable_funcs = {f for f in caps_a["functions"]
                                              if any(v in f.lower() for v in verb_gap)}
                            if len(valuable_funcs) >= 3:
                                gaps.append({
                                    "title": f"Port capabilities from {slug_a} to {slug_b}",
                                    "idea_type": "cross_pollination",
                                    "description": f"{slug_a} has capabilities ({', '.join(sorted(verb_gap)[:5])}) not in {slug_b}: {', '.join(sorted(valuable_funcs)[:5])}",
                                    "rationale": "Same-category projects should share proven capabilities",
                                    "affected_projects": [slug_a, slug_b],
                                    "impact": "medium",
                                    "effort": "high",
                                })
        finally:
            conn.close()

        return gaps[:30]

    def find_enhancement_opportunities(self) -> list[dict]:
        """Find high-frequency patterns not yet extracted, recurring learnings."""
        ideas = []

        # Patterns with high frequency → skill extraction candidates
        patterns = self.db.get_patterns()
        for p in patterns:
            freq = p.get("frequency", 0) or 0
            if freq >= 5:
                projects = p.get("used_by_projects", "[]")
                try:
                    proj_list = json.loads(projects) if isinstance(projects, str) else projects
                except (json.JSONDecodeError, TypeError):
                    proj_list = []
                ideas.append({
                    "title": f"Extract shared pattern: {p['name']}",
                    "idea_type": "enhancement",
                    "description": f"Pattern '{p['name']}' appears {freq} times — extract into reusable shared module",
                    "rationale": f"High-frequency pattern ({freq}x) should be a first-class shared component",
                    "affected_projects": proj_list,
                    "impact": "medium" if freq < 10 else "high",
                    "effort": "low",
                })

        # Recurring learnings → systemic improvement
        conn = self.db._conn()
        try:
            categories = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM learnings GROUP BY category HAVING cnt >= 5 ORDER BY cnt DESC"
            ).fetchall()
        finally:
            conn.close()

        for row in categories:
            cat_data = dict(row)
            ideas.append({
                "title": f"Systemic improvement: {cat_data['category']} ({cat_data['cnt']} learnings)",
                "idea_type": "enhancement",
                "description": f"{cat_data['cnt']} learnings in '{cat_data['category']}' category suggest a systemic issue worth addressing",
                "rationale": "Recurring learnings indicate a pattern that should be codified or automated",
                "affected_projects": [],
                "impact": "medium" if cat_data["cnt"] < 10 else "high",
                "effort": "medium",
            })

        return ideas

    def find_automation_opportunities(self) -> list[dict]:
        """Find repetitive operations that could be automated."""
        ideas = []
        conn = self.db._conn()

        try:
            # Projects with many functions but no scheduled/automated workflows
            projects = self._get_projects_cached()
            for p in projects:
                func_count = p.get("function_count", 0) or 0
                if func_count < 15:
                    continue

                funcs = conn.execute(
                    "SELECT name FROM functions WHERE project_slug = ?", (p["slug"],)
                ).fetchall()
                func_names = [r["name"] for r in funcs]

                has_manual = any(kw in fn.lower() for fn in func_names
                                for kw in ["manual", "run_once", "fix_", "migrate_"])
                has_batch = any(kw in fn.lower() for fn in func_names
                                for kw in ["batch", "schedule", "cron", "daemon", "loop"])

                if has_manual and not has_batch:
                    manual_funcs = [fn for fn in func_names
                                    if any(kw in fn.lower() for kw in ["manual", "run_once", "fix_", "migrate_"])]
                    ideas.append({
                        "title": f"Automate manual operations in {p['slug']}",
                        "idea_type": "automation",
                        "description": f"{p['slug']} has {len(manual_funcs)} manual/one-shot functions but no batch/scheduler: {', '.join(manual_funcs[:5])}",
                        "rationale": "Manual operations should be automated via scheduler or daemon loop",
                        "affected_projects": [p["slug"]],
                        "impact": "medium",
                        "effort": "medium",
                    })
        finally:
            conn.close()

        return ideas

    def generate_new_project_ideas(self) -> list[dict]:
        """Algorithmically discover synergies from project capabilities."""
        ideas = []
        projects = self._get_projects_cached()
        conn = self.db._conn()

        try:
            # Build capability profiles for all projects with sufficient code
            profiles = {}
            for p in projects:
                if (p.get("function_count", 0) or 0) < 5:
                    continue
                caps = self._get_project_capabilities(p["slug"], conn)
                profiles[p["slug"]] = {
                    "category": p.get("category", "uncategorized"),
                    "name": p.get("name", p["slug"]),
                    "caps": caps,
                }

            # Find synergistic pairs: different categories, complementary capabilities
            slugs = list(profiles.keys())
            scored_pairs = []

            for i, slug_a in enumerate(slugs):
                for slug_b in slugs[i + 1:]:
                    prof_a = profiles[slug_a]
                    prof_b = profiles[slug_b]

                    # Same category = not synergy, just duplication
                    if prof_a["category"] == prof_b["category"]:
                        continue

                    verbs_a = prof_a["caps"]["capability_verbs"]
                    verbs_b = prof_b["caps"]["capability_verbs"]

                    # Synergy = complementary capabilities (union is larger than either)
                    shared = verbs_a & verbs_b
                    unique_a = verbs_a - verbs_b
                    unique_b = verbs_b - verbs_a
                    complementary = unique_a | unique_b

                    if len(complementary) < 3 or len(verbs_a) < 2 or len(verbs_b) < 2:
                        continue

                    # Score: complementarity ratio × category compatibility × size
                    complement_ratio = len(complementary) / max(len(verbs_a | verbs_b), 1)
                    cat_key = tuple(sorted([prof_a["category"], prof_b["category"]]))
                    cat_compat = CATEGORY_COMPATIBILITY.get(cat_key, 0.4)
                    size_factor = min((len(verbs_a) + len(verbs_b)) / 10, 2.0)

                    synergy_score = complement_ratio * cat_compat * size_factor

                    if synergy_score >= 0.3:
                        scored_pairs.append({
                            "slug_a": slug_a, "slug_b": slug_b,
                            "name_a": prof_a["name"], "name_b": prof_b["name"],
                            "unique_a": unique_a, "unique_b": unique_b,
                            "shared": shared,
                            "score": synergy_score,
                        })

            # Take top synergies
            scored_pairs.sort(key=lambda x: -x["score"])
            for pair in scored_pairs[:8]:
                combined_caps = sorted(pair["unique_a"] | pair["unique_b"] | pair["shared"])
                ideas.append({
                    "title": f"Integrate {pair['name_a']} + {pair['name_b']}",
                    "idea_type": "new_project",
                    "description": (
                        f"Combine {pair['slug_a']} ({', '.join(sorted(pair['unique_a'])[:3])}) with "
                        f"{pair['slug_b']} ({', '.join(sorted(pair['unique_b'])[:3])}) — "
                        f"synergy score {pair['score']:.2f}"
                    ),
                    "rationale": (
                        f"Complementary capabilities across categories. "
                        f"Combined: {', '.join(combined_caps[:6])}. "
                        f"Already share: {', '.join(sorted(pair['shared'])[:3]) or 'nothing'}"
                    ),
                    "affected_projects": [pair["slug_a"], pair["slug_b"]],
                    "impact": "high" if pair["score"] >= 0.6 else "medium",
                    "effort": "medium",
                })
        finally:
            conn.close()

        return ideas

    def cross_pollinate(self) -> list[dict]:
        """Deep cross-pollination using function signature + behavior analysis."""
        ideas = []
        conn = self.db._conn()

        try:
            # Find functions with similar semantic signatures across different projects
            reusable_patterns = conn.execute("""
                SELECT name, GROUP_CONCAT(DISTINCT project_slug) as projects,
                       COUNT(DISTINCT project_slug) as proj_count
                FROM functions
                WHERE name NOT IN ('main', 'init', 'setup', 'run', 'start', 'stop',
                                  '__init__', '__str__', '__repr__', 'health', 'test',
                                  'get', 'set', 'update', 'delete', 'create', 'close',
                                  'connect', 'process', 'handle')
                  AND name NOT LIKE 'test_%'
                  AND name NOT LIKE '_%'
                  AND length(name) > 5
                GROUP BY name
                HAVING proj_count >= 2
                ORDER BY proj_count DESC
                LIMIT 40
            """).fetchall()

            # Group by shared function names
            shared_capabilities = {}
            for row in reusable_patterns:
                r = dict(row)
                shared_capabilities[r["name"]] = r["projects"].split(",")

            # Build project-pair overlap with function name tracking
            project_overlap = defaultdict(lambda: {"count": 0, "functions": []})
            for func_name, slugs in shared_capabilities.items():
                for i, a in enumerate(slugs):
                    for b in slugs[i + 1:]:
                        key = tuple(sorted([a, b]))
                        project_overlap[key]["count"] += 1
                        if len(project_overlap[key]["functions"]) < 10:
                            project_overlap[key]["functions"].append(func_name)

            # Top overlapping pairs from DIFFERENT categories
            projects_data = {p["slug"]: p for p in self._get_projects_cached()}
            for (slug_a, slug_b), data in sorted(project_overlap.items(), key=lambda x: -x[1]["count"])[:15]:
                if data["count"] < 3:
                    continue
                cat_a = projects_data.get(slug_a, {}).get("category", "")
                cat_b = projects_data.get(slug_b, {}).get("category", "")

                # Cross-category is more interesting than same-category
                if cat_a == cat_b:
                    continue

                ideas.append({
                    "title": f"Cross-pollinate {slug_a} ↔ {slug_b}",
                    "idea_type": "cross_pollination",
                    "description": (
                        f"{slug_a} ({cat_a}) and {slug_b} ({cat_b}) share {data['count']} functions: "
                        f"{', '.join(data['functions'][:5])}"
                    ),
                    "rationale": "Cross-category projects with shared code patterns can create powerful integrations or shared libraries",
                    "affected_projects": [slug_a, slug_b],
                    "impact": "high" if data["count"] >= 6 else "medium",
                    "effort": "medium",
                })

            # Also find projects with high verb-stem overlap across categories
            projects = self._get_projects_cached()
            profiles = {}
            for p in projects:
                if (p.get("function_count", 0) or 0) >= 10:
                    caps = self._get_project_capabilities(p["slug"], conn)
                    profiles[p["slug"]] = {
                        "category": p.get("category", ""),
                        "verbs": caps["capability_verbs"],
                    }

            slug_list = list(profiles.keys())
            for i, sa in enumerate(slug_list):
                for sb in slug_list[i + 1:]:
                    pa, pb = profiles[sa], profiles[sb]
                    if pa["category"] == pb["category"]:
                        continue
                    shared_verbs = pa["verbs"] & pb["verbs"]
                    if len(shared_verbs) >= 3:
                        # Don't duplicate if already found via function names
                        key = tuple(sorted([sa, sb]))
                        if key in project_overlap and project_overlap[key]["count"] >= 3:
                            continue
                        ideas.append({
                            "title": f"Semantic overlap: {sa} ↔ {sb}",
                            "idea_type": "cross_pollination",
                            "description": (
                                f"{sa} and {sb} share {len(shared_verbs)} capability types: "
                                f"{', '.join(sorted(shared_verbs)[:5])}"
                            ),
                            "rationale": "Similar capability profiles across categories suggest reusable patterns or integration opportunities",
                            "affected_projects": [sa, sb],
                            "impact": "medium",
                            "effort": "medium",
                        })
        finally:
            conn.close()

        return ideas[:20]

    def prioritize_ideas(self, ideas: list[dict]) -> list[dict]:
        """Transparent scoring: impact × effort_ease × type_bonus × breadth_factor.

        Score breakdown for any idea:
        - base = IMPACT_SCORES[impact] × EFFORT_EASE[effort]  (range: 1-15)
        - type_adjusted = base × TYPE_BONUSES[idea_type]       (range: 1-21)
        - breadth_bonus = min(affected_projects × 0.2, 1.5)    (range: 0-1.5)
        - raw = type_adjusted + breadth_bonus                   (range: 1-22.5)
        - final = min(10.0, raw / 2.25)                        (normalized to 0-10)
        """
        for idea in ideas:
            impact = self.IMPACT_SCORES.get(
                idea.get("estimated_impact", idea.get("impact", "medium")), 2.0
            )
            effort_ease = self.EFFORT_EASE.get(
                idea.get("estimated_effort", idea.get("effort", "medium")), 2.0
            )
            type_bonus = self.TYPE_BONUSES.get(
                idea.get("idea_type", "enhancement"), 1.0
            )

            base = impact * effort_ease
            type_adjusted = base * type_bonus

            projects = idea.get("affected_projects", [])
            breadth_bonus = min(len(projects) * 0.2, 1.5)

            raw = type_adjusted + breadth_bonus
            # Normalize to 0-10 scale (max theoretical raw ≈ 22.5)
            idea["priority_score"] = round(min(self.MAX_SCORE, raw / 2.25), 2)

        ideas.sort(key=lambda x: -x.get("priority_score", 0))
        return ideas

    def full_ideation_pass(self, evolution_id: int = None) -> dict:
        """Run all idea generation methods, store in DB."""
        results = {
            "feature_gaps": 0,
            "enhancements": 0,
            "automations": 0,
            "new_projects": 0,
            "cross_pollination": 0,
            "total": 0,
        }

        all_ideas = []

        # Feature gaps (same-category comparison)
        log.info("[IdeaEngine] Scanning feature gaps...")
        gaps = self.find_feature_gaps()
        all_ideas.extend(gaps)
        results["feature_gaps"] = len(gaps)

        # Enhancement opportunities (patterns + learnings)
        log.info("[IdeaEngine] Scanning enhancement opportunities...")
        enhancements = self.find_enhancement_opportunities()
        all_ideas.extend(enhancements)
        results["enhancements"] = len(enhancements)

        # Automation opportunities
        log.info("[IdeaEngine] Scanning automation opportunities...")
        automations = self.find_automation_opportunities()
        all_ideas.extend(automations)
        results["automations"] = len(automations)

        # New project ideas (algorithmic synergy detection)
        log.info("[IdeaEngine] Generating synergy-based project ideas...")
        new_projects = self.generate_new_project_ideas()
        all_ideas.extend(new_projects)
        results["new_projects"] = len(new_projects)

        # Cross-pollination (function + behavior analysis)
        log.info("[IdeaEngine] Running cross-pollination analysis...")
        xpoll = self.cross_pollinate()
        all_ideas.extend(xpoll)
        results["cross_pollination"] = len(xpoll)

        # Prioritize all ideas with transparent scoring
        all_ideas = self.prioritize_ideas(all_ideas)

        # Store in DB with evolution_id tracking
        stored = 0
        for idea in all_ideas:
            self.db.add_idea(
                title=idea["title"],
                idea_type=idea.get("idea_type", "enhancement"),
                description=idea.get("description", ""),
                rationale=idea.get("rationale", ""),
                projects=idea.get("affected_projects", []),
                impact=idea.get("impact", idea.get("estimated_impact", "medium")),
                effort=idea.get("effort", idea.get("estimated_effort", "medium")),
                priority_score=idea.get("priority_score", 0),
                evolution_id=evolution_id,
            )
            stored += 1

        results["total"] = len(all_ideas)
        results["stored"] = stored
        log.info(f"[IdeaEngine] Complete: {results['total']} ideas ({stored} stored)")
        return results
