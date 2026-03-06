"""ProfileSentinel -- scores profile quality across 6 criteria (100 points).

Part of the OpenClaw FORGE intelligence layer. Follows the RitualSentinel
pattern: fixed-point rubric with 6 criteria, actionable feedback, and
auto-enhancement when the score falls below a threshold.

Scoring Criteria (100 points total):
    completeness          (0-20)  All fields filled?
    seo_quality           (0-20)  Keywords in bio/description/tagline, proper length?
    brand_consistency     (0-15)  Matches brand identity?
    link_presence         (0-15)  Website, social links present and matching?
    bio_quality           (0-15)  Engaging, not generic, proper length?
    avatar_quality        (0-15)  Has avatar, has banner if supported?

All logic is algorithmic -- zero LLM cost.
"""

from __future__ import annotations

import copy
from typing import Any

from openclaw.models import (
    ProfileContent,
    SentinelScore,
    PlatformConfig,
    QualityGrade,
)
from openclaw.knowledge.platforms import get_platform
from openclaw.knowledge.brand_config import get_brand
from openclaw.knowledge.profile_templates import (
    get_tagline,
    get_bio,
    get_description,
    get_seo_keywords,
)


# ---------------------------------------------------------------------------
# Generic opening phrases we penalize -- signal low-effort profiles
# ---------------------------------------------------------------------------

_GENERIC_OPENERS = [
    "welcome to my",
    "hi there",
    "hello world",
    "this is my",
    "i am a",
    "we are a",
    "just a",
    "my name is",
    "check out my",
]

# Minimum bio lengths for quality scoring (chars)
_BIO_MIN_GOOD = 50
_BIO_MIN_GREAT = 120

# Minimum description lengths
_DESC_MIN_GOOD = 100
_DESC_MIN_GREAT = 300


# =========================================================================== #
#  ProfileSentinel                                                             #
# =========================================================================== #


class ProfileSentinel:
    """Scores profile content across 6 criteria and auto-enhances.

    Given a :class:`ProfileContent`, scores it against a 100-point rubric,
    provides human-readable feedback and enhancement suggestions, and can
    automatically fix common issues when the score falls below a threshold.

    Usage::

        sentinel = ProfileSentinel()
        score = sentinel.score(profile_content)
        print(score.total_score)   # 72.0
        print(score.grade)         # QualityGrade.C
        print(score.feedback)      # ["Bio is too short", ...]

        # Auto-enhance below threshold
        new_score, enhanced = sentinel.score_and_enhance(profile_content, threshold=75.0)
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def score(self, content: ProfileContent) -> SentinelScore:
        """Score profile content across 6 criteria.

        Args:
            content: The profile content to evaluate.

        Returns:
            A SentinelScore with per-criterion scores, total, grade,
            feedback, and enhancement suggestions.
        """
        platform = get_platform(content.platform_id)
        result = SentinelScore(platform_id=content.platform_id)

        result.completeness = self._score_completeness(content, platform)
        result.seo_quality = self._score_seo(content)
        result.brand_consistency = self._score_brand(content)
        result.link_presence = self._score_links(content, platform)
        result.bio_quality = self._score_bio(content, platform)
        result.avatar_quality = self._score_avatar(content, platform)

        result.calculate()
        result.feedback = self._generate_feedback(result, content, platform)
        result.enhancements = self._suggest_enhancements(result, content, platform)
        return result

    def score_and_enhance(
        self,
        content: ProfileContent,
        threshold: float = 70.0,
    ) -> tuple[SentinelScore, ProfileContent]:
        """Score and auto-enhance if below threshold.

        Args:
            content: The profile content to evaluate and potentially enhance.
            threshold: Minimum acceptable score. If the score is below this,
                auto-enhancement is applied.

        Returns:
            A tuple of (SentinelScore, ProfileContent). If the original score
            met the threshold, the original content is returned unchanged.
            Otherwise, the enhanced content and its new score are returned.
        """
        initial_score = self.score(content)
        if initial_score.total_score >= threshold:
            return initial_score, content

        enhanced = self._auto_enhance(content, initial_score)
        new_score = self.score(enhanced)
        return new_score, enhanced

    # ------------------------------------------------------------------ #
    #  Criterion 1: Completeness (0-20)                                    #
    # ------------------------------------------------------------------ #

    def _score_completeness(
        self, content: ProfileContent, platform: PlatformConfig | None
    ) -> float:
        """Score whether all expected fields are populated.

        5 fields checked, 4 points each:
        - username present (4 pts)
        - bio present (4 pts)
        - tagline present (4 pts)
        - description present (4 pts)
        - email present (4 pts)
        """
        score = 0.0

        if content.username:
            score += 4.0
        if content.bio:
            score += 4.0
        if content.tagline:
            score += 4.0
        if content.description:
            score += 4.0
        if content.email:
            score += 4.0

        return min(score, 20.0)

    # ------------------------------------------------------------------ #
    #  Criterion 2: SEO Quality (0-20)                                     #
    # ------------------------------------------------------------------ #

    def _score_seo(self, content: ProfileContent) -> float:
        """Score SEO keyword presence and content quality.

        - Keywords in bio (5 pts)
        - Keywords in description (5 pts)
        - Keywords in tagline (5 pts)
        - Proper content lengths (5 pts)
        """
        score = 0.0
        keywords = content.seo_keywords or []

        if not keywords:
            # No keywords configured -- give partial credit for having content
            if content.bio:
                score += 2.0
            if content.description:
                score += 2.0
            if content.tagline:
                score += 2.0
            return min(score, 20.0)

        # Keywords in bio
        if content.bio:
            bio_lower = content.bio.lower()
            hits = sum(1 for kw in keywords if kw.lower() in bio_lower)
            if hits >= 3:
                score += 5.0
            elif hits >= 2:
                score += 4.0
            elif hits >= 1:
                score += 2.5

        # Keywords in description
        if content.description:
            desc_lower = content.description.lower()
            hits = sum(1 for kw in keywords if kw.lower() in desc_lower)
            if hits >= 3:
                score += 5.0
            elif hits >= 2:
                score += 4.0
            elif hits >= 1:
                score += 2.5

        # Keywords in tagline
        if content.tagline:
            tag_lower = content.tagline.lower()
            hits = sum(1 for kw in keywords if kw.lower() in tag_lower)
            if hits >= 2:
                score += 5.0
            elif hits >= 1:
                score += 3.0

        # Proper lengths (bio >= 50, description >= 100, tagline >= 20)
        length_score = 0.0
        if len(content.bio) >= _BIO_MIN_GOOD:
            length_score += 2.0
        if len(content.description) >= _DESC_MIN_GOOD:
            length_score += 2.0
        if len(content.tagline) >= 20:
            length_score += 1.0
        score += length_score

        return min(score, 20.0)

    # ------------------------------------------------------------------ #
    #  Criterion 3: Brand Consistency (0-15)                               #
    # ------------------------------------------------------------------ #

    def _score_brand(self, content: ProfileContent) -> float:
        """Score how well the profile matches brand identity.

        - Brand name in bio or display_name (5 pts)
        - Consistent voice / tagline matches brand (5 pts)
        - Website URL matches brand website (5 pts)
        """
        score = 0.0
        brand = get_brand()

        brand_name = getattr(brand, "name", "")
        brand_tagline = getattr(brand, "tagline", "")
        brand_website = getattr(brand, "website", "")

        # Brand name present in bio or display_name
        if brand_name:
            name_lower = brand_name.lower()
            if content.display_name and name_lower in content.display_name.lower():
                score += 5.0
            elif content.bio and name_lower in content.bio.lower():
                score += 4.0
            elif content.username and name_lower in content.username.lower():
                score += 2.0

        # Consistent tagline (brand tagline appears or is similar)
        if brand_tagline and content.tagline:
            brand_tag_lower = brand_tagline.lower()
            content_tag_lower = content.tagline.lower()
            # Check if tagline contains key words from brand tagline
            brand_words = set(brand_tag_lower.split())
            content_words = set(content_tag_lower.split())
            overlap = brand_words & content_words
            # Remove very common words
            common = {"the", "a", "an", "and", "or", "for", "to", "in", "of", "is"}
            meaningful_overlap = overlap - common
            if len(meaningful_overlap) >= 3:
                score += 5.0
            elif len(meaningful_overlap) >= 2:
                score += 3.5
            elif len(meaningful_overlap) >= 1:
                score += 2.0
        elif brand_tagline:
            # No content tagline at all
            pass

        # Website URL matches brand
        if brand_website and content.website_url:
            brand_domain = self._extract_domain(brand_website)
            content_domain = self._extract_domain(content.website_url)
            if brand_domain and content_domain and brand_domain == content_domain:
                score += 5.0
            elif content.website_url:
                score += 2.0  # Has a URL, just doesn't match brand
        elif content.website_url:
            score += 3.0  # Has URL, no brand URL to compare

        return min(score, 15.0)

    # ------------------------------------------------------------------ #
    #  Criterion 4: Link Presence (0-15)                                   #
    # ------------------------------------------------------------------ #

    def _score_links(
        self, content: ProfileContent, platform: PlatformConfig | None
    ) -> float:
        """Score link presence and quality.

        - Website URL present (5 pts)
        - Social links count (5 pts): 1 link = 2pt, 2 = 3pt, 3+ = 5pt
        - Links match brand (5 pts)
        """
        score = 0.0
        brand = get_brand()

        # Website URL
        if content.website_url:
            score += 5.0

        # Social links count
        social_count = len(content.social_links) if content.social_links else 0
        if social_count >= 3:
            score += 5.0
        elif social_count == 2:
            score += 3.0
        elif social_count == 1:
            score += 2.0

        # Links match brand social links
        brand_socials = getattr(brand, "social_links", {})
        if brand_socials and content.social_links:
            matching = 0
            for platform_name, url in content.social_links.items():
                brand_url = brand_socials.get(platform_name, "")
                if brand_url and url and self._urls_match(url, brand_url):
                    matching += 1
            if matching >= 3:
                score += 5.0
            elif matching >= 2:
                score += 3.5
            elif matching >= 1:
                score += 2.0
        elif content.social_links:
            score += 2.0  # Has links, no brand links to compare

        return min(score, 15.0)

    # ------------------------------------------------------------------ #
    #  Criterion 5: Bio Quality (0-15)                                     #
    # ------------------------------------------------------------------ #

    def _score_bio(
        self, content: ProfileContent, platform: PlatformConfig | None
    ) -> float:
        """Score bio content quality.

        - Not generic (5 pts)
        - Proper length for platform (5 pts)
        - Engaging opening (5 pts)
        """
        score = 0.0
        bio = content.bio or ""

        if not bio:
            return 0.0

        bio_lower = bio.lower().strip()

        # Not generic: check for generic opening phrases
        is_generic = any(bio_lower.startswith(phrase) for phrase in _GENERIC_OPENERS)
        if not is_generic:
            score += 5.0
        else:
            score += 1.0  # At least has a bio

        # Proper length
        max_len = platform.bio_max_length if platform else 500
        bio_len = len(bio)
        if bio_len >= _BIO_MIN_GREAT and bio_len <= max_len:
            score += 5.0
        elif bio_len >= _BIO_MIN_GOOD and bio_len <= max_len:
            score += 3.5
        elif bio_len > 0 and bio_len <= max_len:
            score += 1.5
        elif bio_len > max_len:
            score += 1.0  # Has content but exceeds limit

        # Engaging opening (starts with action verb, question, or value prop)
        engaging_starters = [
            "build", "creat", "help", "discover", "unlock", "transform",
            "empow", "design", "craft", "automat", "deliver", "provid",
            "we build", "we create", "we help", "we deliver",
            "making", "bringing", "turning",
        ]
        has_engaging_start = any(
            bio_lower.startswith(s) for s in engaging_starters
        )
        # Also count questions as engaging
        if has_engaging_start:
            score += 5.0
        elif "?" in bio[:80]:
            score += 4.0
        elif bio_len >= _BIO_MIN_GOOD:
            score += 2.0  # Long enough to have substance

        return min(score, 15.0)

    # ------------------------------------------------------------------ #
    #  Criterion 6: Avatar Quality (0-15)                                  #
    # ------------------------------------------------------------------ #

    def _score_avatar(
        self, content: ProfileContent, platform: PlatformConfig | None
    ) -> float:
        """Score avatar and banner image presence.

        - Has avatar path (8 pts)
        - Has banner path if platform supports it (7 pts)
        """
        score = 0.0

        # Avatar
        if content.avatar_path:
            score += 8.0

        # Banner
        if platform and platform.allows_banner:
            if content.banner_path:
                score += 7.0
        else:
            # Platform does not support banners -- award full banner points
            score += 7.0

        return min(score, 15.0)

    # ------------------------------------------------------------------ #
    #  Feedback generation                                                 #
    # ------------------------------------------------------------------ #

    def _generate_feedback(
        self,
        result: SentinelScore,
        content: ProfileContent,
        platform: PlatformConfig | None,
    ) -> list[str]:
        """Generate human-readable feedback based on scoring results.

        Args:
            result: The scoring result.
            content: The profile content.
            platform: The platform configuration.

        Returns:
            A list of feedback strings, most critical first.
        """
        feedback: list[str] = []

        # Completeness feedback
        if result.completeness < 12:
            missing = []
            if not content.username:
                missing.append("username")
            if not content.bio:
                missing.append("bio")
            if not content.tagline:
                missing.append("tagline")
            if not content.description:
                missing.append("description")
            if not content.email:
                missing.append("email")
            if missing:
                feedback.append(f"Missing fields: {', '.join(missing)}")

        # SEO feedback
        if result.seo_quality < 10:
            if not content.seo_keywords:
                feedback.append("No SEO keywords defined -- add target keywords for discoverability")
            else:
                feedback.append(
                    "Low keyword presence -- weave target keywords naturally "
                    "into bio, description, and tagline"
                )

        # Brand consistency feedback
        if result.brand_consistency < 8:
            feedback.append(
                "Brand identity not strongly reflected -- ensure brand name "
                "appears in display name or bio"
            )

        # Link feedback
        if result.link_presence < 8:
            if not content.website_url:
                feedback.append("No website URL -- add your main website for credibility")
            if not content.social_links:
                feedback.append("No social links -- add at least 2-3 social profiles")

        # Bio quality feedback
        if result.bio_quality < 8:
            bio = content.bio or ""
            if not bio:
                feedback.append("Bio is empty -- write a compelling bio with your value proposition")
            elif len(bio) < _BIO_MIN_GOOD:
                feedback.append(
                    f"Bio is too short ({len(bio)} chars) -- aim for at least "
                    f"{_BIO_MIN_GOOD} characters"
                )
            else:
                bio_lower = bio.lower()
                if any(bio_lower.startswith(g) for g in _GENERIC_OPENERS):
                    feedback.append(
                        "Bio starts with a generic phrase -- lead with your "
                        "unique value proposition or an action verb"
                    )

        # Avatar feedback
        if result.avatar_quality < 8:
            if not content.avatar_path:
                feedback.append("No avatar image -- upload a branded avatar for trust signals")
            if platform and platform.allows_banner and not content.banner_path:
                feedback.append("No banner image -- upload a banner for a professional look")

        # Grade summary
        if result.grade in (QualityGrade.S, QualityGrade.A):
            feedback.insert(0, f"Excellent profile! Grade: {result.grade.value}")
        elif result.grade == QualityGrade.B:
            feedback.insert(0, f"Good profile with room for improvement. Grade: {result.grade.value}")
        elif result.grade == QualityGrade.C:
            feedback.insert(0, f"Acceptable but needs work. Grade: {result.grade.value}")
        else:
            feedback.insert(0, f"Profile needs significant improvement. Grade: {result.grade.value}")

        return feedback

    # ------------------------------------------------------------------ #
    #  Enhancement suggestions                                             #
    # ------------------------------------------------------------------ #

    def _suggest_enhancements(
        self,
        result: SentinelScore,
        content: ProfileContent,
        platform: PlatformConfig | None,
    ) -> list[str]:
        """Suggest specific enhancements to raise the score.

        Args:
            result: The scoring result.
            content: The profile content.
            platform: The platform configuration.

        Returns:
            A list of enhancement suggestion strings, prioritized by impact.
        """
        enhancements: list[str] = []

        # Highest-impact enhancements first
        if result.completeness < 16:
            enhancements.append(
                "Fill all profile fields (username, bio, tagline, description, email) "
                "-- each adds 4 points to completeness"
            )

        if result.seo_quality < 12:
            enhancements.append(
                "Add target keywords to your bio and description -- "
                "this directly improves discoverability in search"
            )

        if result.brand_consistency < 10:
            brand = get_brand()
            brand_name = getattr(brand, "name", "your brand name")
            enhancements.append(
                f"Include '{brand_name}' in your display name and mention it "
                f"in the first sentence of your bio"
            )

        if result.link_presence < 10:
            enhancements.append(
                "Add your website URL and at least 2 social media links -- "
                "this builds credibility and cross-platform presence"
            )

        if result.bio_quality < 10:
            if content.bio and len(content.bio) < _BIO_MIN_GREAT:
                max_len = platform.bio_max_length if platform else 500
                enhancements.append(
                    f"Expand your bio to {min(_BIO_MIN_GREAT, max_len)} chars -- "
                    f"include what you offer, who you serve, and what makes you unique"
                )
            else:
                enhancements.append(
                    "Start your bio with an action verb or value statement "
                    "instead of a generic greeting"
                )

        if result.avatar_quality < 15:
            if not content.avatar_path:
                enhancements.append("Upload a branded avatar image (logo or headshot)")
            if platform and platform.allows_banner and not content.banner_path:
                enhancements.append("Upload a branded banner image for visual impact")

        return enhancements[:6]

    # ------------------------------------------------------------------ #
    #  Auto-enhance                                                        #
    # ------------------------------------------------------------------ #

    def _auto_enhance(
        self, content: ProfileContent, score: SentinelScore
    ) -> ProfileContent:
        """Automatically fix common profile issues by pulling from templates.

        Creates a deep copy and fills in missing or weak fields from the
        profile template library and brand config.

        Args:
            content: The original profile content.
            score: The scoring result identifying weak areas.

        Returns:
            An enhanced copy of the profile content.
        """
        enhanced = copy.deepcopy(content)
        brand = get_brand()
        platform = get_platform(content.platform_id)
        category = platform.category.value if platform else "digital_product"

        # Fix missing username
        if not enhanced.username:
            username_base = getattr(brand, "username_base", "")
            if username_base:
                max_len = platform.username_max_length if platform else 30
                enhanced.username = username_base[:max_len]

        # Fix missing display name
        if not enhanced.display_name:
            enhanced.display_name = getattr(brand, "name", "")

        # Fix missing email
        if not enhanced.email:
            enhanced.email = getattr(brand, "email", "")

        # Fix missing or weak bio
        if score.bio_quality < 8:
            template_bio = get_bio(category)
            if template_bio:
                max_len = platform.bio_max_length if platform else 500
                enhanced.bio = template_bio[:max_len]

        # Fix missing tagline
        if not enhanced.tagline or score.brand_consistency < 8:
            template_tagline = get_tagline(category)
            if template_tagline:
                max_len = platform.tagline_max_length if platform else 100
                enhanced.tagline = template_tagline[:max_len]

        # Fix missing description
        if not enhanced.description:
            template_desc = get_description(category)
            if template_desc:
                max_len = platform.description_max_length if platform else 2000
                enhanced.description = template_desc[:max_len]

        # Fix missing website
        if not enhanced.website_url:
            enhanced.website_url = getattr(brand, "website", "")

        # Fix missing avatar
        if not enhanced.avatar_path:
            enhanced.avatar_path = getattr(brand, "avatar_path", "")

        # Fix missing banner
        if not enhanced.banner_path and platform and platform.allows_banner:
            enhanced.banner_path = getattr(brand, "banner_path", "")

        # Fix missing social links
        if not enhanced.social_links:
            brand_socials = getattr(brand, "social_links", {})
            if brand_socials and platform:
                max_links = platform.max_links
                enhanced.social_links = dict(
                    list(brand_socials.items())[:max_links]
                )

        # Fix missing SEO keywords
        if not enhanced.seo_keywords:
            enhanced.seo_keywords = get_seo_keywords(category)

        return enhanced

    # ------------------------------------------------------------------ #
    #  Utility helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain from a URL, stripping protocol and www."""
        domain = url.lower().strip()
        for prefix in ("https://", "http://", "www."):
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        # Strip trailing path
        domain = domain.split("/")[0]
        return domain

    @staticmethod
    def _urls_match(url1: str, url2: str) -> bool:
        """Check if two URLs point to the same destination (loose match)."""
        def normalize(u: str) -> str:
            u = u.lower().strip().rstrip("/")
            for prefix in ("https://", "http://", "www."):
                if u.startswith(prefix):
                    u = u[len(prefix):]
            return u

        return normalize(url1) == normalize(url2)
