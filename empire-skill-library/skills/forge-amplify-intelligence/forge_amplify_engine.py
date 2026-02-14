"""
FORGE+AMPLIFY Intelligence Engine — Unified, project-agnostic version.

Refactored from:
  - vision-auditor-project/skills/forge-intelligence/scripts/forge_engine.py
  - vision-auditor-project/skills/amplify-system/scripts/amplify_engine.py

FORGE modules: Scout, Sentinel, Oracle, Smith, Codex
AMPLIFY stages: Enrich, Expand, Fortify, Anticipate, Optimize, Validate
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("forge_amplify")


# ═══════════════════════════════════════════════════════════════════════════════
# FORGE: Five Intelligence Modules
# ═══════════════════════════════════════════════════════════════════════════════


class Scout:
    """Gap detector — finds blind spots in audit configurations."""

    def __init__(self, coverage_map: Dict[str, Any]):
        self.coverage_map = coverage_map

    def analyze_gaps(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze config against coverage map, return gaps."""
        gaps = []
        coverage_score = 0
        total_checks = 0

        for category, levels in self.coverage_map.items():
            for level in ["required", "recommended", "optional"]:
                checks = levels.get(level, [])
                total_checks += len(checks)
                for check in checks:
                    present = self._check_present(config, category, check)
                    if present:
                        weight = {"required": 3, "recommended": 2, "optional": 1}[level]
                        coverage_score += weight
                    elif level == "required":
                        gaps.append({
                            "category": category,
                            "check": check,
                            "level": level,
                            "severity": "high",
                            "suggestion": f"Add {check} check to {category} configuration"
                        })
                    elif level == "recommended":
                        gaps.append({
                            "category": category,
                            "check": check,
                            "level": level,
                            "severity": "medium",
                            "suggestion": f"Consider adding {check} to {category}"
                        })

        max_score = total_checks * 3
        pct = round((coverage_score / max_score * 100) if max_score > 0 else 0, 1)

        return {
            "coverage_score": pct,
            "total_gaps": len(gaps),
            "required_gaps": len([g for g in gaps if g["level"] == "required"]),
            "recommended_gaps": len([g for g in gaps if g["level"] == "recommended"]),
            "gaps": gaps,
            "categories_checked": list(self.coverage_map.keys())
        }

    def _check_present(self, config: Dict, category: str, check: str) -> bool:
        """Check if a specific audit check is configured."""
        cat_config = config.get(category, config.get("checks", {}))
        if isinstance(cat_config, dict):
            return check in cat_config or any(check in str(v) for v in cat_config.values())
        if isinstance(cat_config, list):
            return check in cat_config
        return False


class Sentinel:
    """Prompt optimizer — scores and enhances audit prompts."""

    CRITERIA = {
        "specificity": {"weight": 20, "desc": "Does it name exact elements to check?"},
        "structure": {"weight": 15, "desc": "Does it define output format?"},
        "severity": {"weight": 15, "desc": "Does it define severity levels?"},
        "context": {"weight": 15, "desc": "Does it include site/niche context?"},
        "negative_examples": {"weight": 15, "desc": "Does it specify what NOT to flag?"},
        "actionability": {"weight": 20, "desc": "Does it request fixes, not just issues?"}
    }

    def score_prompt(self, prompt: str) -> Dict[str, Any]:
        """Score a prompt on 6 criteria (0-100)."""
        scores = {}
        total = 0

        # Specificity — check for concrete element references
        specificity_keywords = ["header", "footer", "nav", "button", "image", "font",
                                "color", "spacing", "margin", "padding", "contrast",
                                "alignment", "grid", "card", "link", "form"]
        hits = sum(1 for kw in specificity_keywords if kw.lower() in prompt.lower())
        scores["specificity"] = min(100, hits * 15)

        # Structure — check for output format instructions
        structure_keywords = ["format", "json", "list", "table", "category", "section",
                              "for each", "include", "report", "severity", "score"]
        hits = sum(1 for kw in structure_keywords if kw.lower() in prompt.lower())
        scores["structure"] = min(100, hits * 20)

        # Severity — check for severity framework
        severity_keywords = ["critical", "high", "medium", "low", "priority",
                             "must fix", "should fix", "nice to have", "severity"]
        hits = sum(1 for kw in severity_keywords if kw.lower() in prompt.lower())
        scores["severity"] = min(100, hits * 25)

        # Context — check for site/niche awareness
        context_keywords = ["site", "brand", "niche", "audience", "theme", "dark",
                            "light", "wordpress", "mobile", "responsive"]
        hits = sum(1 for kw in context_keywords if kw.lower() in prompt.lower())
        scores["context"] = min(100, hits * 20)

        # Negative examples — things NOT to flag
        neg_keywords = ["don't flag", "ignore", "not an issue", "acceptable",
                        "skip", "exclude", "false positive", "do not"]
        hits = sum(1 for kw in neg_keywords if kw.lower() in prompt.lower())
        scores["negative_examples"] = min(100, hits * 30)

        # Actionability — does it ask for fixes?
        action_keywords = ["fix", "solution", "recommendation", "css", "html",
                           "code", "change", "update", "modify", "implement"]
        hits = sum(1 for kw in action_keywords if kw.lower() in prompt.lower())
        scores["actionability"] = min(100, hits * 20)

        # Weighted total
        for criterion, data in self.CRITERIA.items():
            total += scores.get(criterion, 0) * data["weight"] / 100

        return {
            "total_score": round(total, 1),
            "max_score": 100,
            "grade": "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D" if total >= 40 else "F",
            "breakdown": {k: {"score": v, "weight": self.CRITERIA[k]["weight"]} for k, v in scores.items()},
            "suggestions": self._generate_suggestions(scores)
        }

    def _generate_suggestions(self, scores: Dict[str, int]) -> List[str]:
        """Generate improvement suggestions for low-scoring criteria."""
        suggestions = []
        if scores.get("specificity", 0) < 60:
            suggestions.append("Add specific UI element names (header, nav, button, card, etc.)")
        if scores.get("structure", 0) < 60:
            suggestions.append("Define expected output format (JSON, categories, severity levels)")
        if scores.get("severity", 0) < 60:
            suggestions.append("Include severity framework (CRITICAL/HIGH/MEDIUM/LOW)")
        if scores.get("context", 0) < 60:
            suggestions.append("Add site context (niche, theme type, audience)")
        if scores.get("negative_examples", 0) < 40:
            suggestions.append("Add negative examples to reduce false positives")
        if scores.get("actionability", 0) < 60:
            suggestions.append("Request specific fixes (CSS/HTML) not just issue descriptions")
        return suggestions


class Oracle:
    """Pattern predictor — predicts where issues will appear."""

    def __init__(self, known_patterns: Dict[str, Any]):
        self.patterns = known_patterns

    def predict(self, site_info: Dict[str, Any], history: List[Dict]) -> List[Dict[str, Any]]:
        """Predict issues based on site characteristics and history."""
        predictions = []
        domain = site_info.get("domain", "")
        characteristics = site_info.get("characteristics", [])

        # Match against known patterns
        all_patterns = {}
        all_patterns.update(self.patterns.get("wordpress_patterns", {}))
        all_patterns.update(self.patterns.get("seo_patterns", {}))

        for pattern_name, pattern in all_patterns.items():
            trigger = pattern.get("trigger", "")
            # Check if trigger matches site characteristics
            if any(trigger in c for c in characteristics) or trigger in domain:
                predictions.append({
                    "pattern": pattern_name,
                    "confidence": pattern.get("confidence", 0.5),
                    "likely_issues": pattern.get("likely_issues", []),
                    "check_first": pattern.get("check_first", []),
                    "based_on": "known_pattern"
                })

        # Check history for recurring patterns
        if history:
            issue_counts: Dict[str, int] = {}
            for audit in history[-10:]:
                for issue in audit.get("issues", []):
                    issue_type = issue.get("type", issue.get("category", "unknown"))
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            for issue_type, count in issue_counts.items():
                if count >= 3:
                    predictions.append({
                        "pattern": f"recurring_{issue_type}",
                        "confidence": min(0.95, 0.5 + count * 0.1),
                        "likely_issues": [issue_type],
                        "check_first": [],
                        "based_on": f"seen {count} times in last {len(history)} audits"
                    })

        # Sort by confidence
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions


class Smith:
    """Auto-fix generator — generates CSS/HTML/content fixes."""

    FIX_TEMPLATES = {
        "contrast_failure": {
            "type": "css",
            "template": "color: {suggested_color}; /* WCAG AA contrast fix */",
            "verification": "Check contrast ratio >= 4.5:1"
        },
        "missing_alt_text": {
            "type": "html_attr",
            "template": 'alt="{generated_alt}"',
            "verification": "Verify alt text describes the image content"
        },
        "heading_skip": {
            "type": "html",
            "template": "Change <{current}> to <{correct}>",
            "verification": "Verify heading hierarchy: h1 > h2 > h3"
        },
        "missing_meta_description": {
            "type": "meta",
            "template": '<meta name="description" content="{generated_description}">',
            "verification": "Verify description is 150-160 characters"
        },
        "mobile_overflow": {
            "type": "css",
            "template": "max-width: 100%; overflow-x: hidden;",
            "verification": "Test on 375px viewport"
        },
        "broken_link": {
            "type": "html",
            "template": "Remove or update href to valid URL",
            "verification": "Verify link returns 200 status"
        },
        "image_not_loading": {
            "type": "html",
            "template": "Update src to valid image URL or add fallback",
            "verification": "Verify image loads and dimensions are correct"
        }
    }

    def __init__(self, custom_templates: Optional[Dict] = None):
        self.templates = {**self.FIX_TEMPLATES}
        if custom_templates:
            self.templates.update(custom_templates)

    def generate_fixes(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fix suggestions for a list of issues."""
        fixes = []
        for issue in issues:
            issue_type = issue.get("type", issue.get("category", ""))
            fix = self._match_fix(issue_type, issue)
            if fix:
                fixes.append({
                    "issue": issue,
                    "fix_type": fix["type"],
                    "fix_code": fix["template"],
                    "verification": fix["verification"],
                    "confidence": fix.get("confidence", 0.85),
                    "auto_applicable": fix.get("confidence", 0.85) >= 0.95
                })
        return fixes

    def _match_fix(self, issue_type: str, issue: Dict) -> Optional[Dict]:
        """Match an issue to a fix template."""
        # Exact match
        if issue_type in self.templates:
            return self.templates[issue_type]
        # Fuzzy match
        for template_key, template in self.templates.items():
            if template_key in issue_type or issue_type in template_key:
                return template
        return None


class Codex:
    """Learning engine — persistent knowledge that improves over time."""

    def __init__(self, knowledge_path: Optional[Path] = None):
        self.knowledge_path = knowledge_path or Path("codex_knowledge.json")
        self.knowledge = self._load_knowledge()

    def _load_knowledge(self) -> Dict[str, Any]:
        """Load knowledge from disk."""
        if self.knowledge_path.exists():
            try:
                with open(self.knowledge_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load codex from {self.knowledge_path}, starting fresh")
        return {
            "total_audits": 0,
            "sites": {},
            "common_issues": {},
            "fix_success_rates": {},
            "cross_site_patterns": [],
            "last_updated": None
        }

    def _save_knowledge(self):
        """Persist knowledge to disk."""
        self.knowledge["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.knowledge_path, "w") as f:
            json.dump(self.knowledge, f, indent=2)

    def ingest_audit(self, site_id: str, results: Dict[str, Any]):
        """Learn from audit results."""
        self.knowledge["total_audits"] += 1

        # Track per-site
        if site_id not in self.knowledge["sites"]:
            self.knowledge["sites"][site_id] = {
                "audit_count": 0,
                "scores": [],
                "recurring_issues": {},
                "last_audit": None
            }
        site_data = self.knowledge["sites"][site_id]
        site_data["audit_count"] += 1
        site_data["last_audit"] = datetime.now(timezone.utc).isoformat()

        score = results.get("score", results.get("overall_score", 0))
        site_data["scores"].append(score)
        # Keep last 50 scores
        site_data["scores"] = site_data["scores"][-50:]

        # Track issues
        for issue in results.get("issues", []):
            issue_type = issue.get("type", issue.get("category", "unknown"))
            self.knowledge["common_issues"][issue_type] = \
                self.knowledge["common_issues"].get(issue_type, 0) + 1
            site_data["recurring_issues"][issue_type] = \
                site_data["recurring_issues"].get(issue_type, 0) + 1

        self._save_knowledge()

    def get_site_trend(self, site_id: str) -> Dict[str, Any]:
        """Get score trend for a site."""
        site_data = self.knowledge.get("sites", {}).get(site_id, {})
        scores = site_data.get("scores", [])
        if len(scores) < 2:
            return {"trend": "insufficient_data", "scores": scores}

        recent = scores[-5:] if len(scores) >= 5 else scores
        older = scores[-10:-5] if len(scores) >= 10 else scores[:len(scores) // 2]

        avg_recent = sum(recent) / len(recent) if recent else 0
        avg_older = sum(older) / len(older) if older else 0
        delta = avg_recent - avg_older

        return {
            "trend": "improving" if delta > 2 else "declining" if delta < -2 else "stable",
            "current_avg": round(avg_recent, 1),
            "previous_avg": round(avg_older, 1),
            "delta": round(delta, 1),
            "audit_count": site_data.get("audit_count", 0),
            "last_audit": site_data.get("last_audit")
        }

    def get_empire_health(self) -> Dict[str, Any]:
        """Get overall empire health summary."""
        sites = self.knowledge.get("sites", {})
        if not sites:
            return {"status": "no_data", "message": "No audits ingested yet"}

        site_summaries = {}
        for site_id, data in sites.items():
            scores = data.get("scores", [])
            site_summaries[site_id] = {
                "audit_count": data.get("audit_count", 0),
                "latest_score": scores[-1] if scores else None,
                "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
                "recurring_issues": dict(sorted(
                    data.get("recurring_issues", {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:5]),
                "last_audit": data.get("last_audit")
            }

        all_scores = [s["latest_score"] for s in site_summaries.values() if s["latest_score"] is not None]
        top_issues = dict(sorted(
            self.knowledge.get("common_issues", {}).items(),
            key=lambda x: x[1], reverse=True
        )[:10])

        return {
            "total_audits": self.knowledge.get("total_audits", 0),
            "sites_tracked": len(sites),
            "empire_avg_score": round(sum(all_scores) / len(all_scores), 1) if all_scores else None,
            "top_issues": top_issues,
            "sites": site_summaries,
            "last_updated": self.knowledge.get("last_updated")
        }


class ForgeIntelligence:
    """Master controller for all five FORGE modules."""

    def __init__(self, coverage_map: Dict, known_patterns: Dict,
                 codex_path: Optional[Path] = None,
                 custom_fix_templates: Optional[Dict] = None):
        self.scout = Scout(coverage_map)
        self.sentinel = Sentinel()
        self.oracle = Oracle(known_patterns)
        self.smith = Smith(custom_fix_templates)
        self.codex = Codex(codex_path)

    def pre_audit_analysis(self, site_info: Dict, config: Dict, prompt: str = "") -> Dict[str, Any]:
        """Run all pre-audit intelligence."""
        gaps = self.scout.analyze_gaps(config)
        predictions = self.oracle.predict(
            site_info,
            self.codex.knowledge.get("sites", {}).get(site_info.get("domain", ""), {}).get("history", [])
        )
        prompt_score = self.sentinel.score_prompt(prompt) if prompt else None

        return {
            "coverage": gaps,
            "predictions": predictions,
            "prompt_score": prompt_score,
            "site_trend": self.codex.get_site_trend(site_info.get("domain", ""))
        }

    def post_audit_enhancement(self, site_id: str, results: Dict) -> Dict[str, Any]:
        """Run all post-audit intelligence."""
        # Learn from results
        self.codex.ingest_audit(site_id, results)

        # Generate fixes
        fixes = self.smith.generate_fixes(results.get("issues", []))

        return {
            "fixes": fixes,
            "auto_fixable": len([f for f in fixes if f.get("auto_applicable")]),
            "manual_fixes": len([f for f in fixes if not f.get("auto_applicable")]),
            "trend": self.codex.get_site_trend(site_id)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AMPLIFY: Six-Stage Enhancement Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SiteProfile:
    """Profile for a specific site/niche."""
    domain: str
    niche: str = "general"
    theme_type: str = "light"
    primary_color: str = "#000000"
    characteristics: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)


@dataclass
class AmplifyRequest:
    """Request to be amplified."""
    url: str = ""
    site_profile: Optional[SiteProfile] = None
    audit_type: str = "visual"
    viewports: List[Dict] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    pre_actions: List[Dict] = field(default_factory=list)
    post_actions: List[Dict] = field(default_factory=list)
    enhanced_prompt: str = ""
    amplified: bool = False


@dataclass
class AmplifyResult:
    """Result of the AMPLIFY pipeline."""
    request: AmplifyRequest
    stages_completed: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    processing_time_ms: float = 0.0
    enrichments: Dict[str, Any] = field(default_factory=dict)
    expansions: Dict[str, Any] = field(default_factory=dict)
    fortifications: Dict[str, Any] = field(default_factory=dict)
    anticipations: Dict[str, Any] = field(default_factory=dict)
    optimizations: Dict[str, Any] = field(default_factory=dict)
    validations: Dict[str, Any] = field(default_factory=dict)


# Default niche intelligence for AMPLIFY ENRICH stage
DEFAULT_NICHE_INTELLIGENCE = {
    "spirituality": {
        "common_themes": ["dark backgrounds", "mystical imagery", "ornate typography"],
        "user_expectations": ["immersive experience", "atmospheric design"],
        "known_pitfalls": ["contrast issues on dark themes", "decorative fonts hard to read"],
        "acceptable_patterns": ["dark theme", "decorative borders", "animated backgrounds"]
    },
    "smart_home": {
        "common_themes": ["clean tech aesthetic", "product imagery", "comparison tables"],
        "user_expectations": ["quick product comparison", "clear specs"],
        "known_pitfalls": ["table overflow on mobile", "product image sizing"],
        "acceptable_patterns": ["tech blue palette", "grid layouts"]
    },
    "ai_tech": {
        "common_themes": ["modern minimal design", "data visualization", "gradient accents"],
        "user_expectations": ["fast loading", "scannable content"],
        "known_pitfalls": ["chart/graph overflow", "code block formatting"],
        "acceptable_patterns": ["dark mode option", "monospace code blocks"]
    },
    "mythology": {
        "common_themes": ["rich imagery", "ornate design", "deep colors"],
        "user_expectations": ["immersive storytelling", "visual richness"],
        "known_pitfalls": ["image-heavy pages slow loading", "ornate fonts illegible"],
        "acceptable_patterns": ["decorative headers", "parchment textures"]
    },
    "lifestyle": {
        "common_themes": ["warm colors", "photography-forward", "approachable typography"],
        "user_expectations": ["friendly, easy navigation", "clear categories"],
        "known_pitfalls": ["hero image sizing", "card grid alignment"],
        "acceptable_patterns": ["rounded corners", "soft shadows", "handwriting accents"]
    },
    "reviews": {
        "common_themes": ["comparison tables", "star ratings", "product cards"],
        "user_expectations": ["quick specs comparison", "clear verdict"],
        "known_pitfalls": ["table responsive issues", "affiliate link density"],
        "acceptable_patterns": ["pros/cons boxes", "comparison tables", "rating widgets"]
    }
}

VIEWPORT_LIBRARY = {
    "mobile_small": {"name": "Mobile Small", "width": 320, "height": 568},
    "mobile": {"name": "Mobile", "width": 375, "height": 812},
    "mobile_large": {"name": "Mobile Large", "width": 428, "height": 926},
    "tablet": {"name": "Tablet", "width": 768, "height": 1024},
    "tablet_landscape": {"name": "Tablet Landscape", "width": 1024, "height": 768},
    "laptop": {"name": "Laptop", "width": 1366, "height": 768},
    "desktop": {"name": "Desktop", "width": 1920, "height": 1080},
    "ultrawide": {"name": "Ultrawide", "width": 2560, "height": 1440},
}


class AmplifyEngine:
    """Six-stage enhancement pipeline for audit requests."""

    def __init__(self, niche_intelligence: Optional[Dict] = None,
                 site_profiles: Optional[Dict[str, SiteProfile]] = None):
        self.niche_data = niche_intelligence or DEFAULT_NICHE_INTELLIGENCE
        self.profiles: Dict[str, SiteProfile] = site_profiles or {}

    def amplify(self, request: AmplifyRequest) -> AmplifyResult:
        """Run the full 6-stage AMPLIFY pipeline."""
        start = time.time()
        result = AmplifyResult(request=request)

        # Stage 1: ENRICH
        result.enrichments = self._enrich(request)
        result.stages_completed.append("ENRICH")

        # Stage 2: EXPAND
        result.expansions = self._expand(request)
        result.stages_completed.append("EXPAND")

        # Stage 3: FORTIFY
        result.fortifications = self._fortify(request)
        result.stages_completed.append("FORTIFY")

        # Stage 4: ANTICIPATE
        result.anticipations = self._anticipate(request)
        result.stages_completed.append("ANTICIPATE")

        # Stage 5: OPTIMIZE
        result.optimizations = self._optimize(request)
        result.stages_completed.append("OPTIMIZE")

        # Stage 6: VALIDATE
        result.validations = self._validate(request)
        result.stages_completed.append("VALIDATE")

        request.amplified = True
        result.processing_time_ms = round((time.time() - start) * 1000, 2)
        result.quality_score = self._calculate_quality_score(result)
        return result

    def _enrich(self, request: AmplifyRequest) -> Dict[str, Any]:
        """Stage 1: Add site-specific context."""
        enrichments = {}

        if request.site_profile:
            enrichments["site_profile"] = {
                "domain": request.site_profile.domain,
                "niche": request.site_profile.niche,
                "theme_type": request.site_profile.theme_type,
                "characteristics": request.site_profile.characteristics
            }
            # Load niche intelligence
            niche = request.site_profile.niche
            if niche in self.niche_data:
                enrichments["niche_intelligence"] = self.niche_data[niche]
        elif request.url:
            domain = urlparse(request.url).netloc.replace("www.", "")
            if domain in self.profiles:
                profile = self.profiles[domain]
                enrichments["site_profile"] = {
                    "domain": profile.domain,
                    "niche": profile.niche,
                    "theme_type": profile.theme_type
                }
                if profile.niche in self.niche_data:
                    enrichments["niche_intelligence"] = self.niche_data[profile.niche]

        return enrichments

    def _expand(self, request: AmplifyRequest) -> Dict[str, Any]:
        """Stage 2: Broaden scope with viewports and edge cases."""
        # Set default viewports if none specified
        if not request.viewports:
            request.viewports = [
                VIEWPORT_LIBRARY["mobile"],
                VIEWPORT_LIBRARY["tablet"],
                VIEWPORT_LIBRARY["desktop"]
            ]

        # Add edge cases based on niche
        edge_cases = list(request.edge_cases)
        if request.site_profile:
            if request.site_profile.theme_type == "dark":
                edge_cases.extend(["contrast_on_dark", "link_visibility_dark", "form_borders_dark"])
            if "ecommerce" in request.site_profile.characteristics:
                edge_cases.extend(["product_grid_alignment", "price_display_overflow"])

        request.edge_cases = edge_cases

        return {
            "viewports": [v.get("name", f"{v['width']}x{v['height']}") for v in request.viewports],
            "edge_cases": edge_cases,
            "viewport_count": len(request.viewports)
        }

    def _fortify(self, request: AmplifyRequest) -> Dict[str, Any]:
        """Stage 3: Error-proof with retries and overlay dismissal."""
        pre_actions = list(request.pre_actions)

        # Add cookie/overlay dismissal
        dismiss_selectors = [
            {"action": "click", "selector": "[class*='cookie'] button[class*='accept']", "optional": True},
            {"action": "click", "selector": "[class*='consent'] button[class*='accept']", "optional": True},
            {"action": "click", "selector": ".modal .close, [class*='popup'] .close", "optional": True},
            {"action": "wait", "selector": "body", "timeout": 3000}
        ]
        pre_actions.extend(dismiss_selectors)
        request.pre_actions = pre_actions

        return {
            "pre_actions": len(pre_actions),
            "retry_config": {"attempts": 3, "backoff_ms": [1000, 3000, 10000]},
            "overlay_dismissal": True,
            "lazy_load_handling": True
        }

    def _anticipate(self, request: AmplifyRequest) -> Dict[str, Any]:
        """Stage 4: Think ahead — predict issues and prepare fixes."""
        anticipated = {}
        if request.site_profile:
            niche = request.site_profile.niche
            niche_data = self.niche_data.get(niche, {})
            anticipated["likely_pitfalls"] = niche_data.get("known_pitfalls", [])
            anticipated["acceptable_patterns"] = niche_data.get("acceptable_patterns", [])
        return anticipated

    def _optimize(self, request: AmplifyRequest) -> Dict[str, Any]:
        """Stage 5: Build the enhanced prompt."""
        prompt = self._build_enhanced_prompt(request)
        request.enhanced_prompt = prompt
        return {
            "prompt_length": len(prompt),
            "has_niche_context": bool(request.site_profile),
            "viewport_coverage": len(request.viewports)
        }

    def _validate(self, request: AmplifyRequest) -> Dict[str, Any]:
        """Stage 6: Final validation before execution."""
        errors = []
        warnings = []

        if not request.url and not request.enhanced_prompt:
            errors.append("No URL or prompt specified")
        if not request.viewports:
            warnings.append("No viewports configured, using defaults")
        if not request.enhanced_prompt:
            warnings.append("No enhanced prompt generated")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _build_enhanced_prompt(self, request: AmplifyRequest) -> str:
        """Build a comprehensive audit prompt with all enrichments."""
        parts = ["You are an expert visual QA auditor. Analyze the following screenshot(s) for issues.\n"]

        # Add site context
        if request.site_profile:
            parts.append(f"SITE: {request.site_profile.domain}")
            parts.append(f"NICHE: {request.site_profile.niche}")
            parts.append(f"THEME: {request.site_profile.theme_type}")
            if request.site_profile.characteristics:
                parts.append(f"CHARACTERISTICS: {', '.join(request.site_profile.characteristics)}")
            parts.append("")

        # Add niche-specific guidance
        niche_data = {}
        if request.site_profile and request.site_profile.niche in self.niche_data:
            niche_data = self.niche_data[request.site_profile.niche]
            if niche_data.get("acceptable_patterns"):
                parts.append(f"ACCEPTABLE PATTERNS (do NOT flag): {', '.join(niche_data['acceptable_patterns'])}")
            if niche_data.get("known_pitfalls"):
                parts.append(f"KNOWN PITFALLS (check carefully): {', '.join(niche_data['known_pitfalls'])}")
            parts.append("")

        # Core audit instructions
        parts.append("CHECK FOR:")
        parts.append("1. LAYOUT: Alignment issues, overflow, spacing inconsistencies, broken grids")
        parts.append("2. TYPOGRAPHY: Font rendering, size hierarchy, readability, truncation")
        parts.append("3. IMAGES: Missing/broken images, wrong aspect ratio, alt text, lazy load failures")
        parts.append("4. INTERACTIVE: Button states, form usability, link styling, hover effects")
        parts.append("5. ACCESSIBILITY: Contrast ratio, focus indicators, screen reader compatibility")
        parts.append("6. RESPONSIVE: Mobile/tablet breakpoint issues, touch target sizes")
        parts.append("")

        # Output format
        parts.append("FOR EACH ISSUE FOUND:")
        parts.append("- ISSUE: Clear description")
        parts.append("- LOCATION: Page region (top/middle/bottom, left/center/right)")
        parts.append("- SEVERITY: CRITICAL / HIGH / MEDIUM / LOW")
        parts.append("- IMPACT: User experience effect")
        parts.append("- FIX: Specific CSS/HTML fix recommendation")
        parts.append("")
        parts.append("END WITH:")
        parts.append("- Overall score: X/100")
        parts.append("- Top 3 priority fixes")

        return "\n".join(parts)

    def _calculate_quality_score(self, result: AmplifyResult) -> float:
        """Calculate quality score for the amplification."""
        score = 0
        score += len(result.stages_completed) * 10  # 60 points max
        if result.enrichments.get("site_profile"):
            score += 5
        if result.enrichments.get("niche_intelligence"):
            score += 5
        viewport_count = len(result.expansions.get("viewports", []))
        score += min(viewport_count * 2, 5)
        edge_case_count = len(result.expansions.get("edge_cases", []))
        score += min(edge_case_count, 5)
        if result.fortifications.get("pre_actions"):
            score += 5
        if result.fortifications.get("retry_config"):
            score += 5
        if not result.validations.get("errors"):
            score += 5
        return min(score, 100)

    def amplify_url(self, url: str, **kwargs) -> AmplifyResult:
        """Convenience: amplify a single URL."""
        req = AmplifyRequest(url=url, **kwargs)
        return self.amplify(req)

    def get_enhanced_prompt(self, url: str) -> str:
        """Get the enhanced prompt for a URL without full pipeline."""
        req = AmplifyRequest(url=url)
        domain = urlparse(url).netloc.replace("www.", "")
        if domain in self.profiles:
            req.site_profile = self.profiles[domain]
        return self._build_enhanced_prompt(req)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENGINE: ForgeAmplifyEngine
# ═══════════════════════════════════════════════════════════════════════════════


class ForgeAmplifyEngine:
    """Unified FORGE+AMPLIFY intelligence engine.

    Combines all 5 FORGE modules and 6 AMPLIFY stages into a single
    project-agnostic engine configurable via JSON config files.

    Usage:
        engine = ForgeAmplifyEngine(
            project_type="wordpress_audit",
            config_dir=Path("configs")
        )
        # Pre-audit intelligence
        intel = engine.pre_audit("witchcraftforbeginners", {...}, "audit prompt")
        # Post-audit learning
        enhanced = engine.post_audit("witchcraftforbeginners", {...})
        # Health overview
        health = engine.health()
    """

    def __init__(self, project_type: str = "wordpress_audit",
                 config_dir: Optional[Path] = None,
                 codex_path: Optional[Path] = None,
                 niche_intelligence: Optional[Dict] = None,
                 site_profiles: Optional[Dict[str, SiteProfile]] = None):
        self.project_type = project_type
        self.config_dir = config_dir or Path(__file__).parent / "configs"

        # Load configs
        coverage_map = self._load_config(f"{project_type}.json", "coverage_map")
        known_patterns = self._load_config("known_patterns.json")

        # Initialize FORGE
        codex_dir = codex_path or (self.config_dir.parent / "data" / "codex_knowledge.json")
        self.forge = ForgeIntelligence(
            coverage_map=coverage_map,
            known_patterns=known_patterns,
            codex_path=codex_dir
        )

        # Initialize AMPLIFY
        self.amplify = AmplifyEngine(
            niche_intelligence=niche_intelligence,
            site_profiles=site_profiles
        )

    def _load_config(self, filename: str, key: Optional[str] = None) -> Dict:
        """Load a JSON config file."""
        path = self.config_dir / filename
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get(key, data) if key else data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading config {path}: {e}")
            return {}

    def pre_audit(self, site_id: str, config: Dict, prompt: str = "") -> Dict[str, Any]:
        """Run pre-audit intelligence (FORGE + AMPLIFY)."""
        site_info = {"domain": site_id, "characteristics": ["wordpress"]}
        forge_result = self.forge.pre_audit_analysis(site_info, config, prompt)

        # AMPLIFY the request if we have a URL
        url = config.get("url", f"https://{site_id}.com")
        amplify_result = self.amplify.amplify_url(url)

        return {
            "forge": forge_result,
            "amplify": {
                "stages_completed": amplify_result.stages_completed,
                "quality_score": amplify_result.quality_score,
                "enhanced_prompt": amplify_result.request.enhanced_prompt,
                "viewports": [v.get("name", str(v)) for v in amplify_result.request.viewports],
                "edge_cases": amplify_result.request.edge_cases
            }
        }

    def post_audit(self, site_id: str, results: Dict) -> Dict[str, Any]:
        """Run post-audit intelligence (FORGE learning + fix generation)."""
        return self.forge.post_audit_enhancement(site_id, results)

    def health(self) -> Dict[str, Any]:
        """Get empire-wide health from CODEX."""
        return self.forge.codex.get_empire_health()

    def predict(self, site_id: str) -> List[Dict[str, Any]]:
        """Get ORACLE predictions for a site."""
        site_info = {"domain": site_id, "characteristics": ["wordpress"]}
        history = self.forge.codex.knowledge.get("sites", {}).get(site_id, {}).get("history", [])
        return self.forge.oracle.predict(site_info, history)

    def score_prompt(self, prompt: str) -> Dict[str, Any]:
        """Score a prompt using SENTINEL."""
        return self.forge.sentinel.score_prompt(prompt)

    def learn(self, site_id: str, results: Dict):
        """Ingest audit results into CODEX."""
        self.forge.codex.ingest_audit(site_id, results)

    def generate_fixes(self, issues: List[Dict]) -> List[Dict[str, Any]]:
        """Generate fixes for issues using SMITH."""
        return self.forge.smith.generate_fixes(issues)

    def get_enhanced_prompt(self, url: str) -> str:
        """Get an AMPLIFY-enhanced prompt for a URL."""
        return self.amplify.get_enhanced_prompt(url)
