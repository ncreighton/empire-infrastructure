"""AMPLIFY Pipeline — 6-stage profile optimization for signup plans.

Stages: Enrich → Expand → Fortify → Anticipate → Optimize → Validate
Pattern: videoforge-engine/videoforge/amplify/amplify_pipeline.py
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from openclaw.knowledge.brand_config import get_brand
from openclaw.knowledge.platforms import get_platform
from openclaw.knowledge.profile_templates import (
    get_bio,
    get_description,
    get_seo_keywords,
    get_tagline,
    get_username,
    BIOS,
    DESCRIPTIONS,
    SEO_KEYWORDS,
    TAGLINES,
    USERNAME_PATTERNS,
)
from openclaw.models import (
    AmplifyResult,
    CaptchaType,
    PlatformConfig,
    ProfileContent,
    SignupComplexity,
    SignupPlan,
)

logger = logging.getLogger(__name__)

# Words to avoid in professional profiles
FORBIDDEN_WORDS = [
    "hack", "crack", "exploit", "cheat", "spam", "scam",
    "guaranteed income", "get rich quick", "make money fast",
    "mlm", "pyramid", "clickbait",
]

# TOS-sensitive terms by platform category
TOS_SENSITIVE = {
    "ai_marketplace": ["scraping", "unlimited", "bypass"],
    "digital_product": ["resell rights", "plr", "white label"],
    "education": ["certification", "accredited", "degree"],
    "prompt_marketplace": ["jailbreak", "bypass safety"],
}


class AmplifyPipeline:
    """6-stage optimization pipeline for signup plans and profile content."""

    def amplify(self, plan: SignupPlan) -> AmplifyResult:
        """Run all 6 AMPLIFY stages on a signup plan.

        Each stage mutates the plan's stage dicts directly.
        Returns AmplifyResult with quality score and readiness.
        """
        result = AmplifyResult(plan=plan, stages_completed=0)

        try:
            self._stage_enrich(plan)
            result.stages_completed += 1

            self._stage_expand(plan)
            result.stages_completed += 1

            self._stage_fortify(plan)
            result.stages_completed += 1

            self._stage_anticipate(plan)
            result.stages_completed += 1

            self._stage_optimize(plan)
            result.stages_completed += 1

            self._stage_validate(plan)
            result.stages_completed += 1

        except Exception as e:
            logger.error(f"AMPLIFY failed at stage {result.stages_completed + 1}: {e}")
            result.issues.append(str(e))

        result.quality_score = self._calculate_quality_score(plan)
        result.ready = (
            result.stages_completed == 6
            and result.quality_score >= 70
            and plan.validations.get("all_passed", False)
        )

        result.stage_details = {
            "enrichments": plan.enrichments,
            "expansions": plan.expansions,
            "fortifications": plan.fortifications,
            "anticipations": plan.anticipations,
            "optimizations": plan.optimizations,
            "validations": plan.validations,
        }

        logger.info(
            f"AMPLIFY complete for {plan.platform_id}: "
            f"score={result.quality_score:.0f}, "
            f"stages={result.stages_completed}/6, "
            f"ready={result.ready}"
        )
        return result

    # ─── Stage 1: ENRICH ──────────────────────────────────────────────────

    def _stage_enrich(self, plan: SignupPlan) -> None:
        """Inject brand context, SEO keywords, and platform-specific context."""
        platform = get_platform(plan.platform_id)
        brand = get_brand()
        category = platform.category.value if platform else "general"

        plan.enrichments = {
            "brand_name": brand.name,
            "brand_voice": brand.brand_voice,
            "brand_website": brand.website,
            "brand_tagline": brand.tagline,
            "category": category,
            "seo_keywords": get_seo_keywords(category, count=10),
            "platform_specific": {
                "monetization_potential": platform.monetization_potential if platform else 5,
                "audience_size": platform.audience_size if platform else 5,
                "bio_max_length": platform.bio_max_length if platform else 500,
                "description_max_length": platform.description_max_length if platform else 2000,
                "allows_links": platform.allows_links if platform else True,
            },
            "social_links": {
                k: v for k, v in brand.social_links.items() if v
            },
            "enriched_at": datetime.now().isoformat(),
        }

    # ─── Stage 2: EXPAND ─────────────────────────────────────────────────

    def _stage_expand(self, plan: SignupPlan) -> None:
        """Generate variant bios, taglines, descriptions, and usernames."""
        category = plan.enrichments.get("category", "general")
        brand = get_brand()

        kwargs = {"brand_name": brand.name}

        bio_variants = []
        tagline_variants = []
        description_variants = []
        username_variants = []

        for _ in range(3):
            bio_variants.append(get_bio(category, **kwargs))
            tagline_variants.append(get_tagline(category, **kwargs))
            description_variants.append(get_description(category, **kwargs))
            username_variants.append(get_username(brand.username_base))

        # Deduplicate
        bio_variants = list(dict.fromkeys(bio_variants))
        tagline_variants = list(dict.fromkeys(tagline_variants))
        username_variants = list(dict.fromkeys(username_variants))

        plan.expansions = {
            "bio_variants": bio_variants,
            "tagline_variants": tagline_variants,
            "description_variants": description_variants,
            "username_variants": username_variants,
            "expanded_at": datetime.now().isoformat(),
        }

    # ─── Stage 3: FORTIFY ────────────────────────────────────────────────

    def _stage_fortify(self, plan: SignupPlan) -> None:
        """Validate character limits, forbidden words, TOS compliance."""
        platform = get_platform(plan.platform_id)
        checks_passed = []
        warnings = []

        content = plan.profile_content
        if not content:
            plan.fortifications = {
                "checks_passed": [],
                "warnings": ["No profile content to fortify"],
                "tos_safe": True,
                "forbidden_word_safe": True,
            }
            return

        # Check character limits
        if platform:
            if content.bio and len(content.bio) > platform.bio_max_length:
                warnings.append(
                    f"Bio exceeds limit: {len(content.bio)}/{platform.bio_max_length}"
                )
            else:
                checks_passed.append("bio_length_ok")

            if content.tagline and len(content.tagline) > platform.tagline_max_length:
                warnings.append(
                    f"Tagline exceeds limit: {len(content.tagline)}/{platform.tagline_max_length}"
                )
            else:
                checks_passed.append("tagline_length_ok")

            if content.description and len(content.description) > platform.description_max_length:
                warnings.append(
                    f"Description exceeds limit: "
                    f"{len(content.description)}/{platform.description_max_length}"
                )
            else:
                checks_passed.append("description_length_ok")

            if content.username and len(content.username) > platform.username_max_length:
                warnings.append(
                    f"Username exceeds limit: "
                    f"{len(content.username)}/{platform.username_max_length}"
                )
            else:
                checks_passed.append("username_length_ok")

        # Check forbidden words
        all_text = " ".join(filter(None, [
            content.bio, content.tagline, content.description,
        ])).lower()
        forbidden_found = [w for w in FORBIDDEN_WORDS if w in all_text]
        if forbidden_found:
            warnings.append(f"Forbidden words found: {', '.join(forbidden_found)}")
        else:
            checks_passed.append("no_forbidden_words")

        # Check TOS-sensitive terms
        category = plan.enrichments.get("category", "")
        sensitive_terms = TOS_SENSITIVE.get(category, [])
        tos_violations = [t for t in sensitive_terms if t in all_text]
        if tos_violations:
            warnings.append(f"TOS-sensitive terms: {', '.join(tos_violations)}")
        else:
            checks_passed.append("tos_compliant")

        plan.fortifications = {
            "checks_passed": checks_passed,
            "warnings": warnings,
            "forbidden_word_safe": not forbidden_found,
            "tos_safe": not tos_violations,
            "fortified_at": datetime.now().isoformat(),
        }

    # ─── Stage 4: ANTICIPATE ─────────────────────────────────────────────

    def _stage_anticipate(self, plan: SignupPlan) -> None:
        """Predict CAPTCHA likelihood, verification risks, timing issues."""
        platform = get_platform(plan.platform_id)
        potential_issues = []
        preparation_checklist = []

        if platform:
            # CAPTCHA anticipation
            if platform.captcha_type != CaptchaType.NONE:
                potential_issues.append({
                    "type": "captcha",
                    "severity": "medium",
                    "detail": f"Expect {platform.captcha_type.value} CAPTCHA",
                })
                preparation_checklist.append("Ensure 2Captcha API key is configured")

            # Phone verification
            if platform.requires_phone_verification:
                potential_issues.append({
                    "type": "phone_verification",
                    "severity": "high",
                    "detail": "Phone verification required — may need human intervention",
                })
                preparation_checklist.append("Have phone number ready for SMS verification")

            # Email verification
            if platform.requires_email_verification:
                potential_issues.append({
                    "type": "email_verification",
                    "severity": "low",
                    "detail": "Email verification required — check inbox after signup",
                })
                preparation_checklist.append("Ensure email inbox is accessible")

            # Waitlist
            if platform.has_waitlist:
                potential_issues.append({
                    "type": "waitlist",
                    "severity": "high",
                    "detail": "Platform has waitlist — signup may not be immediate",
                })

            # Complexity-based risks
            if platform.complexity == SignupComplexity.COMPLEX:
                potential_issues.append({
                    "type": "complex_flow",
                    "severity": "medium",
                    "detail": "Complex signup flow — higher failure probability",
                })
            elif platform.complexity == SignupComplexity.MANUAL_ONLY:
                potential_issues.append({
                    "type": "manual_required",
                    "severity": "critical",
                    "detail": "Manual-only signup — cannot fully automate",
                })

            # Known quirks
            for quirk in platform.known_quirks:
                potential_issues.append({
                    "type": "quirk",
                    "severity": "low",
                    "detail": quirk,
                })

        # Time estimation
        estimated_seconds = sum(
            s.timeout_seconds for s in plan.steps
        )

        plan.anticipations = {
            "potential_issues": potential_issues,
            "preparation_checklist": preparation_checklist,
            "estimated_duration_seconds": estimated_seconds,
            "risk_level": self._assess_risk_level(potential_issues),
            "automation_confidence": self._assess_confidence(platform),
            "anticipated_at": datetime.now().isoformat(),
        }

    def _assess_risk_level(self, issues: list[dict]) -> str:
        """Assess overall risk level from anticipated issues."""
        severities = [i.get("severity", "low") for i in issues]
        if "critical" in severities:
            return "critical"
        if severities.count("high") >= 2:
            return "high"
        if "high" in severities:
            return "medium"
        if "medium" in severities:
            return "low"
        return "minimal"

    def _assess_confidence(self, platform: PlatformConfig | None) -> float:
        """Assess automation confidence (0-100)."""
        if not platform:
            return 50.0
        score = 100.0
        if platform.captcha_type != CaptchaType.NONE:
            score -= 15
        if platform.requires_phone_verification:
            score -= 25
        if platform.requires_email_verification:
            score -= 5
        if platform.has_waitlist:
            score -= 20
        if platform.complexity == SignupComplexity.COMPLEX:
            score -= 15
        elif platform.complexity == SignupComplexity.MANUAL_ONLY:
            score -= 40
        score -= len(platform.known_quirks) * 3
        return max(0, score)

    # ─── Stage 5: OPTIMIZE ───────────────────────────────────────────────

    def _stage_optimize(self, plan: SignupPlan) -> None:
        """Optimize profile for platform discovery algorithms."""
        platform = get_platform(plan.platform_id)
        content = plan.profile_content
        optimizations = {}

        if content and platform:
            # SEO optimization
            keywords = plan.enrichments.get("seo_keywords", [])
            bio_keywords = [k for k in keywords if content.bio and k.lower() in content.bio.lower()]
            desc_keywords = [k for k in keywords if content.description and k.lower() in content.description.lower()]

            optimizations["seo_coverage"] = {
                "bio_keywords_found": len(bio_keywords),
                "desc_keywords_found": len(desc_keywords),
                "total_keywords": len(keywords),
                "bio_coverage_pct": (
                    len(bio_keywords) / len(keywords) * 100 if keywords else 0
                ),
            }

            # Link optimization
            link_count = len([v for v in content.social_links.values() if v])
            optimizations["link_optimization"] = {
                "links_filled": link_count,
                "max_links": platform.max_links,
                "has_website": bool(content.website_url),
            }

            # Content completeness
            fields_filled = sum(1 for v in [
                content.username, content.display_name, content.bio,
                content.tagline, content.description, content.website_url,
                content.avatar_path,
            ] if v)
            optimizations["completeness"] = {
                "fields_filled": fields_filled,
                "total_fields": 7,
                "pct": fields_filled / 7 * 100,
            }

        plan.optimizations = {
            **optimizations,
            "optimized_at": datetime.now().isoformat(),
        }

    # ─── Stage 6: VALIDATE ───────────────────────────────────────────────

    def _stage_validate(self, plan: SignupPlan) -> None:
        """Final validation — all content ready, no blockers."""
        checks = []
        issues = []

        content = plan.profile_content

        # Content checks
        if content:
            if content.username:
                checks.append("has_username")
            else:
                issues.append("Missing username")

            if content.email:
                checks.append("has_email")
            else:
                issues.append("Missing email")

            if content.bio:
                checks.append("has_bio")
            else:
                issues.append("Missing bio")

            if content.display_name:
                checks.append("has_display_name")
            else:
                issues.append("Missing display name")
        else:
            issues.append("No profile content generated")

        # Plan checks
        if plan.steps:
            checks.append("has_steps")
        else:
            issues.append("No signup steps planned")

        # Fortification checks
        if plan.fortifications.get("forbidden_word_safe", True):
            checks.append("no_forbidden_words")
        else:
            issues.append("Contains forbidden words")

        if plan.fortifications.get("tos_safe", True):
            checks.append("tos_compliant")
        else:
            issues.append("Contains TOS-sensitive terms")

        # Warnings from fortification
        warnings = plan.fortifications.get("warnings", [])
        if warnings:
            issues.extend(warnings)

        # Risk checks
        risk = plan.anticipations.get("risk_level", "minimal")
        if risk in ("critical",):
            issues.append(f"Risk level is {risk}")
        elif risk in ("high",):
            checks.append("risk_acknowledged")

        all_passed = len(issues) == 0

        plan.validations = {
            "checks": checks,
            "issues": issues,
            "all_passed": all_passed,
            "ready_to_execute": all_passed and len(checks) >= 3,
            "validated_at": datetime.now().isoformat(),
        }

    # ─── Quality Score ────────────────────────────────────────────────────

    def _calculate_quality_score(self, plan: SignupPlan) -> float:
        """Calculate composite quality score (0-100)."""
        score = 0.0

        # Stage completion: 10 points each (60 max)
        if plan.enrichments:
            score += 10
        if plan.expansions:
            score += 10
        if plan.fortifications:
            score += 10
        if plan.anticipations:
            score += 10
        if plan.optimizations:
            score += 10
        if plan.validations:
            score += 10

        # Enrichment depth (+5)
        if len(plan.enrichments.get("seo_keywords", [])) >= 5:
            score += 3
        if plan.enrichments.get("social_links"):
            score += 2

        # Expansion breadth (+5)
        if len(plan.expansions.get("bio_variants", [])) >= 2:
            score += 3
        if len(plan.expansions.get("username_variants", [])) >= 2:
            score += 2

        # Fortification safety (+5)
        if plan.fortifications.get("forbidden_word_safe"):
            score += 2
        if plan.fortifications.get("tos_safe"):
            score += 3

        # Anticipation preparedness (+5)
        confidence = plan.anticipations.get("automation_confidence", 0)
        if confidence >= 70:
            score += 5
        elif confidence >= 50:
            score += 3
        elif confidence >= 30:
            score += 1

        # Optimization efficiency (+5)
        completeness = plan.optimizations.get("completeness", {})
        if completeness.get("pct", 0) >= 80:
            score += 5
        elif completeness.get("pct", 0) >= 60:
            score += 3

        # Validation completeness (+15)
        if plan.validations.get("all_passed"):
            score += 10
        checks = plan.validations.get("checks", [])
        if len(checks) >= 5:
            score += 5
        elif len(checks) >= 3:
            score += 3

        return min(100.0, score)
