"""MarketOracle -- prioritizes platforms by monetization potential, audience fit, effort.

Part of the OpenClaw FORGE intelligence layer. Follows the MoonOracle pattern:
ranking and recommendation engine with composite scoring.

Scoring formula (0-100 composite):
    monetization_potential * 4.0   (40% weight)
    audience_size * 2.5            (25% weight)
    seo_value * 2.0                (20% weight)
    ease_score * 1.5               (15% weight, derived from complexity)

All logic is algorithmic -- zero LLM cost.
"""

from __future__ import annotations

from openclaw.models import (
    PlatformConfig,
    OracleRecommendation,
    OraclePriority,
    PlatformCategory,
    SignupComplexity,
)
from typing import Any

from openclaw.knowledge.platforms import PLATFORMS


# ---------------------------------------------------------------------------
# Complexity penalty map: higher = harder = lower ease score
# ---------------------------------------------------------------------------

_COMPLEXITY_PENALTY: dict[SignupComplexity, float] = {
    SignupComplexity.TRIVIAL: 0.0,
    SignupComplexity.SIMPLE: 2.0,
    SignupComplexity.MODERATE: 4.0,
    SignupComplexity.COMPLEX: 7.0,
    SignupComplexity.MANUAL_ONLY: 10.0,
}

# Category labels for human-readable reasoning
_CATEGORY_LABELS: dict[PlatformCategory, str] = {
    PlatformCategory.AI_MARKETPLACE: "AI marketplace",
    PlatformCategory.WORKFLOW_MARKETPLACE: "workflow marketplace",
    PlatformCategory.DIGITAL_PRODUCT: "digital product platform",
    PlatformCategory.EDUCATION: "education platform",
    PlatformCategory.PROMPT_MARKETPLACE: "prompt marketplace",
    PlatformCategory.THREE_D_MODELS: "3D model marketplace",
    PlatformCategory.CODE_REPOSITORY: "code repository",
    PlatformCategory.SOCIAL_PLATFORM: "social platform",
}


# =========================================================================== #
#  MarketOracle                                                                #
# =========================================================================== #


class MarketOracle:
    """Prioritizes platforms by monetization potential, audience fit, and effort.

    Produces ranked :class:`OracleRecommendation` objects with composite
    scores, priority labels, and human-readable reasoning explaining why
    each platform was ranked where it was.

    Usage::

        oracle = MarketOracle()
        ranked = oracle.prioritize_platforms(completed={"gumroad", "etsy"})
        print(ranked[0].platform_name)  # Best next platform
        print(ranked[0].reasoning)      # Why it's recommended

        # Or just get the single best next action
        rec = oracle.recommend_next(completed={"gumroad"})
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def prioritize_platforms(
        self,
        completed: set[str] | None = None,
        categories: set[PlatformCategory] | None = None,
        min_monetization: int = 0,
        max_complexity: SignupComplexity | None = None,
    ) -> list[OracleRecommendation]:
        """Rank all platforms by composite score, excluding completed ones.

        Args:
            completed: Set of platform IDs already completed (excluded).
            categories: If provided, only consider platforms in these categories.
            min_monetization: Minimum monetization_potential (1-10) to include.
            max_complexity: Maximum signup complexity to include.

        Returns:
            A list of OracleRecommendation objects sorted by score descending,
            each with an assigned ``recommended_order`` (1-based).
        """
        completed = completed or set()
        recommendations: list[OracleRecommendation] = []

        for platform_id, platform in PLATFORMS.items():
            # Skip completed
            if platform_id in completed:
                continue

            # Filter by category
            if categories and platform.category not in categories:
                continue

            # Filter by monetization
            if platform.monetization_potential < min_monetization:
                continue

            # Filter by complexity
            if max_complexity is not None:
                max_penalty = _COMPLEXITY_PENALTY.get(max_complexity, 10.0)
                platform_penalty = _COMPLEXITY_PENALTY.get(platform.complexity, 10.0)
                if platform_penalty > max_penalty:
                    continue

            composite, reasoning = self._calculate_score(platform)
            priority = self._assign_priority(composite)

            rec = OracleRecommendation(
                platform_id=platform_id,
                platform_name=platform.name,
                category=platform.category,
                priority=priority,
                score=composite,
                monetization_score=platform.monetization_potential * 4.0,
                audience_score=platform.audience_size * 2.5,
                effort_score=_COMPLEXITY_PENALTY.get(platform.complexity, 5.0),
                seo_score=platform.seo_value * 2.0,
                reasoning=reasoning,
            )
            recommendations.append(rec)

        # Sort by composite score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)

        # Assign recommended order
        for i, rec in enumerate(recommendations):
            rec.recommended_order = i + 1

        return recommendations

    def recommend_next(
        self,
        completed: set[str] | None = None,
        categories: set[PlatformCategory] | None = None,
    ) -> OracleRecommendation | None:
        """Get the single best next platform to sign up for.

        Args:
            completed: Set of platform IDs already completed.
            categories: Optional category filter.

        Returns:
            The highest-scoring OracleRecommendation, or ``None`` if all
            platforms are completed or excluded.
        """
        ranked = self.prioritize_platforms(
            completed=completed, categories=categories
        )
        return ranked[0] if ranked else None

    def get_category_summary(
        self, completed: set[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Summarize platform coverage and opportunity by category.

        Returns:
            A dict keyed by category value with counts, avg scores, and
            lists of platform IDs.
        """
        completed = completed or set()
        summary: dict[str, dict[str, Any]] = {}

        for platform_id, platform in PLATFORMS.items():
            cat = platform.category.value
            if cat not in summary:
                summary[cat] = {
                    "total": 0,
                    "completed": 0,
                    "remaining": 0,
                    "avg_monetization": 0.0,
                    "platforms": [],
                    "completed_ids": [],
                    "remaining_ids": [],
                }

            entry = summary[cat]
            entry["total"] += 1
            entry["platforms"].append(platform_id)
            entry["avg_monetization"] += platform.monetization_potential

            if platform_id in completed:
                entry["completed"] += 1
                entry["completed_ids"].append(platform_id)
            else:
                entry["remaining"] += 1
                entry["remaining_ids"].append(platform_id)

        # Calculate averages
        for cat, entry in summary.items():
            if entry["total"] > 0:
                entry["avg_monetization"] = round(
                    entry["avg_monetization"] / entry["total"], 1
                )

        return summary

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _calculate_score(
        self, platform: PlatformConfig
    ) -> tuple[float, str]:
        """Calculate composite score and generate reasoning string.

        Formula:
            monetization_potential * 4.0    (max 40)
            + audience_size * 2.5           (max 25)
            + seo_value * 2.0               (max 20)
            + (10 - complexity_penalty) * 1.5 (max 15)

        Args:
            platform: The platform configuration.

        Returns:
            A tuple of (composite_score, reasoning_string).
        """
        # Component scores
        monetization = platform.monetization_potential * 4.0
        audience = platform.audience_size * 2.5
        seo = platform.seo_value * 2.0

        penalty = _COMPLEXITY_PENALTY.get(platform.complexity, 5.0)
        ease = (10.0 - penalty) * 1.5

        composite = round(monetization + audience + seo + ease, 1)

        # Build reasoning
        cat_label = _CATEGORY_LABELS.get(
            platform.category, platform.category.value
        )
        reasons: list[str] = []

        # Monetization reasoning
        if platform.monetization_potential >= 8:
            reasons.append(f"excellent monetization potential ({platform.monetization_potential}/10)")
        elif platform.monetization_potential >= 6:
            reasons.append(f"good monetization potential ({platform.monetization_potential}/10)")
        elif platform.monetization_potential >= 4:
            reasons.append(f"moderate monetization potential ({platform.monetization_potential}/10)")
        else:
            reasons.append(f"limited monetization potential ({platform.monetization_potential}/10)")

        # Audience reasoning
        if platform.audience_size >= 8:
            reasons.append(f"very large built-in audience ({platform.audience_size}/10)")
        elif platform.audience_size >= 6:
            reasons.append(f"solid audience base ({platform.audience_size}/10)")
        elif platform.audience_size >= 4:
            reasons.append(f"moderate audience ({platform.audience_size}/10)")
        else:
            reasons.append(f"smaller niche audience ({platform.audience_size}/10)")

        # SEO reasoning
        if platform.seo_value >= 7:
            reasons.append(f"strong SEO backlink value ({platform.seo_value}/10)")
        elif platform.seo_value >= 5:
            reasons.append(f"decent SEO value ({platform.seo_value}/10)")

        # Complexity reasoning
        if platform.complexity == SignupComplexity.TRIVIAL:
            reasons.append("trivial signup (email + password only)")
        elif platform.complexity == SignupComplexity.SIMPLE:
            reasons.append("simple signup process")
        elif platform.complexity == SignupComplexity.MODERATE:
            reasons.append("moderate signup complexity (may include CAPTCHA)")
        elif platform.complexity == SignupComplexity.COMPLEX:
            reasons.append("complex signup (phone/manual verification)")
        else:
            reasons.append("manual-only signup (requires human intervention)")

        reasoning = (
            f"{platform.name} is a {cat_label} with {', '.join(reasons)}. "
            f"Composite score: {composite}/100."
        )

        return composite, reasoning

    def _assign_priority(self, score: float) -> OraclePriority:
        """Assign a priority level based on composite score.

        Args:
            score: The composite score (0-100).

        Returns:
            An OraclePriority enum member.
        """
        if score >= 80:
            return OraclePriority.CRITICAL
        elif score >= 60:
            return OraclePriority.HIGH
        elif score >= 40:
            return OraclePriority.MEDIUM
        elif score >= 20:
            return OraclePriority.LOW
        else:
            return OraclePriority.SKIP
