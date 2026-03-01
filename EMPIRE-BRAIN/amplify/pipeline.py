"""AMPLIFY Pipeline — 6-Stage Intelligence Enhancement

Takes any Brain output and amplifies it through 6 stages of enhancement.
Ensures we never backtrack, always move forward, always get smarter.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB


class AmplifyPipeline:
    """6-stage enhancement pipeline for Brain intelligence."""

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def amplify(self, data: dict, context: str = "") -> dict:
        """Run data through full 6-stage AMPLIFY pipeline."""
        result = {
            "original": data,
            "context": context,
            "stages_completed": [],
            "amplified_at": datetime.now().isoformat(),
            "quality_score": 0,
        }

        # Stage 1: ENRICH
        enriched = self._enrich(data, context)
        result["enriched"] = enriched
        result["stages_completed"].append("enrich")

        # Stage 2: EXPAND
        expanded = self._expand(data, enriched)
        result["expanded"] = expanded
        result["stages_completed"].append("expand")

        # Stage 3: FORTIFY
        fortified = self._fortify(data, enriched, expanded)
        result["fortified"] = fortified
        result["stages_completed"].append("fortify")

        # Stage 4: ANTICIPATE
        anticipated = self._anticipate(data, enriched, expanded)
        result["anticipated"] = anticipated
        result["stages_completed"].append("anticipate")

        # Stage 5: OPTIMIZE
        optimized = self._optimize(data, enriched, expanded)
        result["optimized"] = optimized
        result["stages_completed"].append("optimize")

        # Stage 6: VALIDATE
        validated = self._validate(result)
        result["validation"] = validated
        result["quality_score"] = validated.get("score", 0)
        result["stages_completed"].append("validate")

        self.db.emit_event("amplify.completed", {
            "context": context[:100],
            "quality_score": result["quality_score"],
            "stages": 6,
        })

        return result

    def amplify_quick(self, data: dict, context: str = "") -> dict:
        """Quick 3-stage amplify (Enrich + Fortify + Validate)."""
        result = {
            "original": data,
            "context": context,
            "stages_completed": [],
            "amplified_at": datetime.now().isoformat(),
        }
        result["enriched"] = self._enrich(data, context)
        result["stages_completed"].append("enrich")
        result["fortified"] = self._fortify(data, result["enriched"], {})
        result["stages_completed"].append("fortify")
        result["validation"] = self._validate(result)
        result["quality_score"] = result["validation"].get("score", 0)
        result["stages_completed"].append("validate")
        return result

    # --- Stage 1: ENRICH ---
    def _enrich(self, data: dict, context: str) -> dict:
        """Add knowledge graph context, learnings, and relevant patterns."""
        enrichment = {
            "relevant_learnings": [],
            "relevant_patterns": [],
            "related_projects": [],
            "knowledge_context": [],
        }

        # Search for relevant learnings
        search_terms = self._extract_search_terms(data, context)
        for term in search_terms[:5]:
            learnings = self.db.search_learnings(term, limit=3)
            for l in learnings:
                enrichment["relevant_learnings"].append({
                    "content": l["content"][:200],
                    "category": l.get("category"),
                    "confidence": l.get("confidence", 0),
                })

        # Find related patterns
        patterns = self.db.get_patterns()
        data_str = json.dumps(data).lower()
        for p in patterns:
            if p.get("name", "").lower() in data_str or any(
                term.lower() in p.get("description", "").lower() for term in search_terms
            ):
                enrichment["relevant_patterns"].append({
                    "name": p["name"],
                    "type": p.get("pattern_type"),
                    "frequency": p.get("frequency", 0),
                })

        # Find related projects
        search = self.db.search(context or json.dumps(data)[:200])
        enrichment["related_projects"] = search.get("projects", [])[:5]

        return enrichment

    # --- Stage 2: EXPAND ---
    def _expand(self, data: dict, enriched: dict) -> dict:
        """Cross-reference across all projects for connections and insights."""
        expansion = {
            "cross_references": [],
            "similar_implementations": [],
            "integration_points": [],
        }

        # Find similar functions/endpoints across projects
        search_terms = self._extract_search_terms(data, "")
        for term in search_terms[:3]:
            results = self.db.search(term)
            if results.get("functions"):
                expansion["similar_implementations"].extend(results["functions"][:3])
            if results.get("endpoints"):
                expansion["integration_points"].extend(results["endpoints"][:3])

        return expansion

    # --- Stage 3: FORTIFY ---
    def _fortify(self, data: dict, enriched: dict, expanded: dict) -> dict:
        """Validate against accumulated learnings and known anti-patterns."""
        fortification = {
            "warnings": [],
            "anti_patterns_detected": [],
            "best_practices_applied": [],
            "confidence_boost": 0,
        }

        # Check against known anti-patterns
        anti_patterns = self.db.get_patterns(pattern_type="anti_pattern")
        data_str = json.dumps(data).lower()
        for ap in anti_patterns:
            if ap.get("name", "").lower() in data_str:
                fortification["anti_patterns_detected"].append({
                    "pattern": ap["name"],
                    "description": ap.get("description", ""),
                    "action": "Review and remediate",
                })
                fortification["warnings"].append(f"Anti-pattern detected: {ap['name']}")

        # Check against known gotchas
        gotchas = self.db.search_learnings("gotcha", limit=5)
        for g in gotchas:
            if any(term in g["content"].lower() for term in self._extract_search_terms(data, "")):
                fortification["warnings"].append(f"Known gotcha: {g['content'][:100]}")

        # Apply learnings as confidence boost
        if enriched.get("relevant_learnings"):
            fortification["confidence_boost"] = min(len(enriched["relevant_learnings"]) * 5, 25)
            fortification["best_practices_applied"] = [
                l["content"][:100] for l in enriched["relevant_learnings"]
                if l.get("confidence", 0) > 0.7
            ]

        return fortification

    # --- Stage 4: ANTICIPATE ---
    def _anticipate(self, data: dict, enriched: dict, expanded: dict) -> dict:
        """Predict impacts, risks, and consequences."""
        anticipation = {
            "potential_impacts": [],
            "risk_factors": [],
            "downstream_effects": [],
        }

        # Check which projects could be affected
        related = enriched.get("related_projects", [])
        for proj in related:
            anticipation["downstream_effects"].append({
                "project": proj.get("slug", proj.get("name", "")),
                "effect": "May need updates based on this change",
            })

        # Check integration points
        for ep in expanded.get("integration_points", []):
            anticipation["potential_impacts"].append({
                "type": "api_dependency",
                "detail": f"Endpoint {ep.get('method', '')} {ep.get('path', '')} in {ep.get('project_slug', '')}",
            })

        return anticipation

    # --- Stage 5: OPTIMIZE ---
    def _optimize(self, data: dict, enriched: dict, expanded: dict) -> dict:
        """Find efficiency improvements and cost savings."""
        optimization = {
            "suggestions": [],
            "cost_savings": [],
            "reusable_components": [],
        }

        # Find reusable code/solutions
        search_terms = self._extract_search_terms(data, "")
        for term in search_terms[:3]:
            conn = self.db._conn()
            solutions = conn.execute(
                "SELECT problem, solution, times_reused FROM code_solutions WHERE problem LIKE ? LIMIT 3",
                (f"%{term}%",)
            ).fetchall()
            conn.close()
            for s in solutions:
                optimization["reusable_components"].append({
                    "problem": s["problem"][:100],
                    "solution_preview": s["solution"][:100],
                    "times_reused": s["times_reused"],
                })

        # Check for shared implementations
        for impl in expanded.get("similar_implementations", []):
            optimization["suggestions"].append(
                f"Similar function '{impl.get('name', '')}' exists in {impl.get('project_slug', '')} — consider reuse"
            )

        return optimization

    # --- Stage 6: VALIDATE ---
    def _validate(self, result: dict) -> dict:
        """Score quality and completeness."""
        score = 50  # Base score

        # +10 for each enrichment found
        enriched = result.get("enriched", {})
        score += min(len(enriched.get("relevant_learnings", [])) * 5, 15)
        score += min(len(enriched.get("relevant_patterns", [])) * 5, 10)

        # +5 for fortification confidence boost
        fortified = result.get("fortified", {})
        score += fortified.get("confidence_boost", 0)

        # -10 for each warning
        score -= len(fortified.get("warnings", [])) * 5

        # -15 for anti-patterns
        score -= len(fortified.get("anti_patterns_detected", [])) * 15

        # +5 for reusable components found
        optimized = result.get("optimized", {})
        score += min(len(optimized.get("reusable_components", [])) * 5, 10)

        score = max(0, min(100, score))

        return {
            "score": score,
            "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F",
            "stages_passed": len(result.get("stages_completed", [])),
            "warnings_count": len(fortified.get("warnings", [])),
            "enrichments_count": len(enriched.get("relevant_learnings", [])),
            "timestamp": datetime.now().isoformat(),
        }

    # --- Helpers ---
    def _extract_search_terms(self, data: dict, context: str) -> list[str]:
        """Extract meaningful search terms from data and context."""
        terms = set()
        text = f"{context} {json.dumps(data)}"

        # Extract quoted strings
        import re
        quoted = re.findall(r'"([^"]+)"', text)
        terms.update(w for w in quoted if 3 < len(w) < 50)

        # Extract key names
        if isinstance(data, dict):
            for key in data:
                if isinstance(key, str) and len(key) > 2:
                    terms.add(key)

        # Extract from context
        words = context.split()
        terms.update(w for w in words if len(w) > 4 and w.isalpha())

        return list(terms)[:10]
