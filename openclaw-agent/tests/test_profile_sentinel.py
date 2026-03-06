"""Tests for openclaw/forge/profile_sentinel.py — ProfileSentinel scoring."""

import pytest

from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.models import ProfileContent, SentinelScore, QualityGrade


@pytest.fixture
def sentinel():
    return ProfileSentinel()


def _make_complete_profile() -> ProfileContent:
    """Build a high-quality profile for scoring."""
    return ProfileContent(
        platform_id="gumroad",
        username="openclawtools",
        display_name="OpenClaw",
        email="test@openclaw.dev",
        bio=(
            "Building AI agents and automation tools that solve real problems. "
            "Every tool is battle-tested in production and documented with examples. "
            "OpenClaw delivers AI tools, workflow automation, and digital products."
        ),
        tagline="AI agents & automation tools | OpenClaw",
        description=(
            "OpenClaw creates AI-powered tools and workflow automation for creators "
            "and developers. Every product is production-tested, well-documented, "
            "and designed to save you hours of work. We build AI tools, agents, "
            "and integrations that people actually use."
        ),
        website_url="https://openclaw.dev",
        avatar_path="/tmp/avatar.png",
        banner_path="/tmp/banner.png",
        social_links={"github": "https://github.com/openclaw", "twitter": "https://twitter.com/openclaw", "linkedin": "https://linkedin.com/in/openclaw"},
        seo_keywords=["AI tools", "automation", "agents", "workflow automation", "digital products"],
    )


def _make_empty_profile() -> ProfileContent:
    """Build a minimal empty profile."""
    return ProfileContent(platform_id="gumroad")


class TestProfileSentinel:
    def test_score_complete_profile_is_high(self, sentinel):
        profile = _make_complete_profile()
        score = sentinel.score(profile)
        assert isinstance(score, SentinelScore)
        assert score.total_score >= 60, f"Expected >= 60, got {score.total_score}"
        assert score.grade in (QualityGrade.S, QualityGrade.A, QualityGrade.B, QualityGrade.C)

    def test_score_empty_profile_is_low(self, sentinel):
        profile = _make_empty_profile()
        score = sentinel.score(profile)
        assert isinstance(score, SentinelScore)
        assert score.total_score < 30
        assert score.grade in (QualityGrade.D, QualityGrade.F)

    def test_all_criteria_within_ranges(self, sentinel):
        profile = _make_complete_profile()
        score = sentinel.score(profile)
        assert 0 <= score.completeness <= 20
        assert 0 <= score.seo_quality <= 20
        assert 0 <= score.brand_consistency <= 15
        assert 0 <= score.link_presence <= 15
        assert 0 <= score.bio_quality <= 15
        assert 0 <= score.avatar_quality <= 15

    def test_criteria_sum_equals_total(self, sentinel):
        profile = _make_complete_profile()
        score = sentinel.score(profile)
        expected = (
            score.completeness + score.seo_quality + score.brand_consistency
            + score.link_presence + score.bio_quality + score.avatar_quality
        )
        assert abs(score.total_score - expected) < 0.01

    def test_feedback_is_populated(self, sentinel):
        profile = _make_complete_profile()
        score = sentinel.score(profile)
        assert isinstance(score.feedback, list)
        assert len(score.feedback) >= 1  # At least the grade summary

    def test_enhancements_for_weak_profile(self, sentinel):
        profile = _make_empty_profile()
        score = sentinel.score(profile)
        assert isinstance(score.enhancements, list)
        assert len(score.enhancements) >= 1

    def test_score_and_enhance_above_threshold(self, sentinel):
        """If score is already above threshold, content is returned unchanged."""
        profile = _make_complete_profile()
        score, content = sentinel.score_and_enhance(profile, threshold=30.0)
        assert isinstance(score, SentinelScore)
        assert content.platform_id == profile.platform_id

    def test_score_and_enhance_below_threshold(self, sentinel):
        """If score is below threshold, auto-enhancement should improve it."""
        profile = _make_empty_profile()
        initial_score = sentinel.score(profile)
        new_score, enhanced = sentinel.score_and_enhance(profile, threshold=70.0)
        assert isinstance(new_score, SentinelScore)
        # Enhanced profile should have more content filled in
        assert enhanced.bio or enhanced.tagline or enhanced.username
        # Score should generally improve
        assert new_score.total_score >= initial_score.total_score

    def test_enhance_fills_missing_fields(self, sentinel):
        """Auto-enhance should fill bio, tagline, username from templates."""
        profile = _make_empty_profile()
        _, enhanced = sentinel.score_and_enhance(profile, threshold=99.0)
        # At least some fields should be filled
        filled_count = sum(1 for v in [
            enhanced.username, enhanced.bio, enhanced.tagline,
            enhanced.display_name
        ] if v)
        assert filled_count >= 2

    def test_complete_profile_all_criteria_nonzero(self, sentinel):
        profile = _make_complete_profile()
        score = sentinel.score(profile)
        assert score.completeness > 0
        assert score.seo_quality > 0
        assert score.bio_quality > 0
        assert score.avatar_quality > 0

    def test_empty_profile_all_criteria_low(self, sentinel):
        profile = _make_empty_profile()
        score = sentinel.score(profile)
        assert score.completeness == 0
        assert score.seo_quality == 0
        assert score.bio_quality == 0

    def test_generic_bio_penalized(self, sentinel):
        """Bio starting with a generic phrase should score lower on bio_quality."""
        generic_profile = ProfileContent(
            platform_id="gumroad",
            username="test",
            email="test@test.com",
            bio="Welcome to my profile! I am here to sell things.",
            tagline="My stuff",
            description="Stuff I sell.",
        )
        good_profile = ProfileContent(
            platform_id="gumroad",
            username="test",
            email="test@test.com",
            bio="Building AI tools that automate real workflows and save teams hours every week.",
            tagline="AI tools for teams",
            description="We create AI tools.",
        )
        generic_score = sentinel.score(generic_profile)
        good_score = sentinel.score(good_profile)
        assert good_score.bio_quality >= generic_score.bio_quality
