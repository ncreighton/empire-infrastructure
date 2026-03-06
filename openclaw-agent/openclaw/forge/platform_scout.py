"""PlatformScout -- analyzes platforms for signup readiness and complexity.

Part of the OpenClaw FORGE intelligence layer. Follows the SpellScout pattern:
take a platform ID, pull its configuration, analyze complexity, build a
readiness checklist, assess risks, and generate practical tips.

All logic is algorithmic -- zero LLM cost.
"""

from __future__ import annotations

import os
from typing import Any

from openclaw.models import (
    PlatformConfig,
    ScoutResult,
    SignupComplexity,
    CaptchaType,
)
from openclaw.knowledge.platforms import get_platform
from openclaw.knowledge.brand_config import get_brand


# ---------------------------------------------------------------------------
# Risk keywords by category
# ---------------------------------------------------------------------------

_CAPTCHA_RISK_LABELS: dict[CaptchaType, str] = {
    CaptchaType.RECAPTCHA_V2: "reCAPTCHA v2 requires image selection -- may need solving service",
    CaptchaType.RECAPTCHA_V3: "reCAPTCHA v3 scores behavior -- use realistic mouse movement and delays",
    CaptchaType.HCAPTCHA: "hCaptcha is increasingly difficult for automation -- may need 2Captcha fallback",
    CaptchaType.TURNSTILE: "Cloudflare Turnstile checks browser fingerprint -- use undetected-chromedriver",
    CaptchaType.FUNCAPTCHA: "FunCAPTCHA is complex -- will likely require human or solving service",
    CaptchaType.IMAGE_CHALLENGE: "Custom image challenge -- manual solving may be needed",
    CaptchaType.UNKNOWN: "Unknown CAPTCHA type -- scout manually before automating",
}

_COMPLEXITY_MINUTES: dict[SignupComplexity, tuple[int, int]] = {
    SignupComplexity.TRIVIAL: (1, 3),
    SignupComplexity.SIMPLE: (3, 5),
    SignupComplexity.MODERATE: (5, 10),
    SignupComplexity.COMPLEX: (10, 20),
    SignupComplexity.MANUAL_ONLY: (15, 30),
}


# =========================================================================== #
#  PlatformScout                                                               #
# =========================================================================== #


class PlatformScout:
    """Analyzes platforms for signup readiness and complexity.

    Given a platform ID, produces a :class:`ScoutResult` with:
    - Signup complexity and estimated time
    - Required and optional field lists
    - Readiness checklist (brand assets vs. platform requirements)
    - Risk assessment (CAPTCHAs, phone verification, waitlists)
    - Practical tips for successful signup

    Usage::

        scout = PlatformScout()
        result = scout.analyze("gumroad")
        print(result.complexity)         # SignupComplexity.SIMPLE
        print(result.completeness_score) # 85.0
        print(result.risks)              # ["reCAPTCHA v2 requires ..."]
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(self, platform_id: str) -> ScoutResult:
        """Full analysis: complexity, readiness checklist, risks, tips.

        Args:
            platform_id: The platform identifier (e.g., "gumroad").

        Returns:
            A fully populated ScoutResult dataclass.

        Raises:
            ValueError: If the platform is not found in the knowledge base.
        """
        platform = get_platform(platform_id)
        if not platform:
            raise ValueError(f"Unknown platform: {platform_id}")

        brand = get_brand()
        required = self._get_required_fields(platform)
        optional = self._get_optional_fields(platform)
        checklist = self._build_readiness_checklist(platform, brand)
        risks = self._assess_risks(platform)
        tips = self._generate_tips(platform)
        completeness = self._score_readiness(checklist)

        return ScoutResult(
            platform_id=platform_id,
            complexity=platform.complexity,
            estimated_minutes=platform.estimated_signup_minutes,
            captcha_type=platform.captcha_type,
            required_fields=required,
            optional_fields=optional,
            readiness_checklist=checklist,
            risks=risks,
            tips=tips,
            completeness_score=completeness,
        )

    def analyze_batch(self, platform_ids: list[str]) -> list[ScoutResult]:
        """Analyze multiple platforms and return sorted by readiness score.

        Args:
            platform_ids: List of platform identifiers.

        Returns:
            A list of ScoutResults sorted by completeness_score descending.
        """
        results = []
        for pid in platform_ids:
            try:
                results.append(self.analyze(pid))
            except ValueError:
                continue
        results.sort(key=lambda r: r.completeness_score, reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  Field extraction                                                    #
    # ------------------------------------------------------------------ #

    def _get_required_fields(self, platform: PlatformConfig) -> list[str]:
        """Extract names of required fields from platform config.

        Args:
            platform: The platform configuration.

        Returns:
            A list of field names that are required for signup.
        """
        return [f.name for f in platform.fields if f.required]

    def _get_optional_fields(self, platform: PlatformConfig) -> list[str]:
        """Extract names of optional fields from platform config.

        Args:
            platform: The platform configuration.

        Returns:
            A list of field names that are optional during signup.
        """
        return [f.name for f in platform.fields if not f.required]

    # ------------------------------------------------------------------ #
    #  Readiness checklist                                                 #
    # ------------------------------------------------------------------ #

    def _build_readiness_checklist(
        self, platform: PlatformConfig, brand: Any
    ) -> list[dict[str, Any]]:
        """Build a readiness checklist comparing brand assets to platform needs.

        Each item is a dict with keys: "item" (str), "ready" (bool), "note" (str).

        Args:
            platform: The platform configuration.
            brand: The BrandIdentity from brand_config.

        Returns:
            A list of checklist item dicts.
        """
        checklist: list[dict[str, Any]] = []

        # 1. Email address
        has_email = bool(getattr(brand, "email", ""))
        checklist.append({
            "item": "Email address",
            "ready": has_email,
            "note": "Ready" if has_email else "Brand email not configured",
        })

        # 2. Username
        has_username = bool(getattr(brand, "username_base", ""))
        checklist.append({
            "item": "Username",
            "ready": has_username,
            "note": "Ready" if has_username else "No username_base in brand config",
        })

        # 3. Avatar image
        has_avatar = False
        if platform.allows_avatar:
            avatar_path = getattr(brand, "avatar_path", "")
            has_avatar = bool(avatar_path) and os.path.exists(avatar_path)
            checklist.append({
                "item": "Avatar image",
                "ready": has_avatar,
                "note": (
                    "Ready" if has_avatar
                    else f"Avatar file missing: {avatar_path}" if avatar_path
                    else "No avatar_path in brand config"
                ),
            })

        # 4. Banner image (if platform supports it)
        if platform.allows_banner:
            banner_path = getattr(brand, "banner_path", "")
            has_banner = bool(banner_path) and os.path.exists(banner_path)
            checklist.append({
                "item": "Banner image",
                "ready": has_banner,
                "note": (
                    "Ready" if has_banner
                    else f"Banner file missing: {banner_path}" if banner_path
                    else "No banner_path in brand config"
                ),
            })

        # 5. Website URL
        has_website = bool(getattr(brand, "website", ""))
        if platform.allows_links:
            checklist.append({
                "item": "Website URL",
                "ready": has_website,
                "note": "Ready" if has_website else "No website in brand config",
            })

        # 6. Brand name / display name
        has_name = bool(getattr(brand, "name", ""))
        checklist.append({
            "item": "Display name",
            "ready": has_name,
            "note": "Ready" if has_name else "No brand name configured",
        })

        # 7. Tagline
        has_tagline = bool(getattr(brand, "tagline", ""))
        if platform.tagline_max_length > 0:
            checklist.append({
                "item": "Tagline",
                "ready": has_tagline,
                "note": (
                    f"Ready (max {platform.tagline_max_length} chars)"
                    if has_tagline
                    else "No tagline in brand config"
                ),
            })

        # 8. Social links
        has_socials = bool(getattr(brand, "social_links", {}))
        if platform.allows_links and platform.max_links > 1:
            checklist.append({
                "item": "Social links",
                "ready": has_socials,
                "note": (
                    f"Ready ({len(getattr(brand, 'social_links', {}))} links, "
                    f"max {platform.max_links})"
                    if has_socials
                    else "No social_links in brand config"
                ),
            })

        # 9. SEO keywords
        has_keywords = bool(getattr(brand, "seo_keywords", []))
        checklist.append({
            "item": "SEO keywords",
            "ready": has_keywords,
            "note": (
                f"Ready ({len(getattr(brand, 'seo_keywords', []))} keywords)"
                if has_keywords
                else "No seo_keywords in brand config"
            ),
        })

        # 10. Required form fields coverage
        required_fields = self._get_required_fields(platform)
        standard_fields = {"email", "username", "password", "name", "display_name"}
        non_standard = [f for f in required_fields if f.lower() not in standard_fields]
        if non_standard:
            checklist.append({
                "item": f"Non-standard required fields: {', '.join(non_standard)}",
                "ready": False,
                "note": "These fields may need manual mapping or custom_fields",
            })

        return checklist

    # ------------------------------------------------------------------ #
    #  Risk assessment                                                     #
    # ------------------------------------------------------------------ #

    def _assess_risks(self, platform: PlatformConfig) -> list[str]:
        """Assess risks and blockers for automated signup.

        Args:
            platform: The platform configuration.

        Returns:
            A list of risk description strings.
        """
        risks: list[str] = []

        # CAPTCHA risk
        if platform.captcha_type != CaptchaType.NONE:
            label = _CAPTCHA_RISK_LABELS.get(
                platform.captcha_type,
                f"CAPTCHA type '{platform.captcha_type.value}' detected",
            )
            risks.append(label)

        # Phone verification
        if platform.requires_phone_verification:
            risks.append(
                "Phone verification required -- need a valid phone number "
                "or SMS verification service"
            )

        # Email verification
        if platform.requires_email_verification:
            risks.append(
                "Email verification required -- ensure inbox access is "
                "automated or monitored"
            )

        # Waitlist
        if platform.has_waitlist:
            risks.append(
                "Platform uses a waitlist -- signup may not grant immediate "
                "access; expect delays of days to weeks"
            )

        # High complexity
        if platform.complexity == SignupComplexity.MANUAL_ONLY:
            risks.append(
                "Manual-only signup -- automation is not recommended; "
                "requires human intervention"
            )
        elif platform.complexity == SignupComplexity.COMPLEX:
            risks.append(
                "Complex signup flow -- expect multi-step process with "
                "potential anti-bot measures"
            )

        # Known quirks
        for quirk in platform.known_quirks:
            risks.append(f"Known quirk: {quirk}")

        return risks

    # ------------------------------------------------------------------ #
    #  Tips                                                                #
    # ------------------------------------------------------------------ #

    def _generate_tips(self, platform: PlatformConfig) -> list[str]:
        """Generate practical tips for successful signup.

        Args:
            platform: The platform configuration.

        Returns:
            A list of actionable tip strings.
        """
        tips: list[str] = []

        # OAuth tip
        if platform.has_oauth and platform.oauth_providers:
            providers = ", ".join(platform.oauth_providers)
            tips.append(
                f"OAuth available ({providers}) -- using OAuth can skip "
                f"CAPTCHA and email verification on many platforms"
            )

        # Username tip
        if platform.username_max_length < 20:
            tips.append(
                f"Username is limited to {platform.username_max_length} chars "
                f"-- prepare a short variant of your brand name"
            )

        # Bio length tip
        if platform.bio_max_length > 0:
            if platform.bio_max_length <= 160:
                tips.append(
                    f"Bio is very short ({platform.bio_max_length} chars) -- "
                    f"focus on your core value proposition only"
                )
            elif platform.bio_max_length <= 300:
                tips.append(
                    f"Bio allows {platform.bio_max_length} chars -- include "
                    f"your value proposition and one key link"
                )

        # Banner tip
        if platform.allows_banner:
            tips.append(
                "Platform supports banner images -- upload a branded banner "
                "for a professional profile appearance"
            )

        # Links tip
        if platform.allows_links and platform.max_links >= 3:
            tips.append(
                f"Up to {platform.max_links} links allowed -- include your "
                f"website, social profiles, and top product page"
            )
        elif platform.allows_links and platform.max_links == 1:
            tips.append(
                "Only 1 link allowed -- use your main website URL or a "
                "link-in-bio page"
            )

        # Timing tip based on complexity
        lo, hi = _COMPLEXITY_MINUTES.get(
            platform.complexity, (5, 10)
        )
        tips.append(f"Expected signup time: {lo}-{hi} minutes")

        # Monetization tip
        if platform.monetization_potential >= 8:
            tips.append(
                "High monetization potential -- prioritize completing "
                "the full profile and uploading sample products quickly"
            )
        elif platform.monetization_potential >= 6:
            tips.append(
                "Good monetization potential -- set up a complete profile "
                "and at least one product or listing"
            )

        # SEO tip
        if platform.seo_value >= 7:
            tips.append(
                "Strong SEO value -- use your target keywords in bio, "
                "tagline, and description for backlink equity"
            )

        return tips

    # ------------------------------------------------------------------ #
    #  Readiness scoring                                                   #
    # ------------------------------------------------------------------ #

    def _score_readiness(self, checklist: list[dict[str, Any]]) -> float:
        """Calculate percentage of checklist items marked ready.

        Args:
            checklist: The readiness checklist from _build_readiness_checklist.

        Returns:
            A float from 0.0 to 100.0 representing readiness percentage.
        """
        if not checklist:
            return 0.0

        ready_count = sum(1 for item in checklist if item["ready"])
        return round((ready_count / len(checklist)) * 100, 1)
