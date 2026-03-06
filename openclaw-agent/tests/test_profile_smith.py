"""Tests for openclaw/forge/profile_smith.py — ProfileSmith content generation."""

import pytest

from openclaw.forge.profile_smith import ProfileSmith
from openclaw.models import ProfileContent
from openclaw.knowledge.platforms import get_platform


@pytest.fixture
def smith():
    return ProfileSmith()


class TestProfileSmith:
    def test_generate_profile_returns_content(self, smith):
        profile = smith.generate_profile("gumroad")
        assert isinstance(profile, ProfileContent)
        assert profile.platform_id == "gumroad"

    def test_all_text_fields_populated(self, smith):
        profile = smith.generate_profile("gumroad")
        assert profile.username, "username should be populated"
        assert profile.bio, "bio should be populated"
        assert profile.generated_at is not None

    def test_unknown_platform_raises(self, smith):
        with pytest.raises(ValueError, match="Unknown platform"):
            smith.generate_profile("nonexistent_xyz_789")

    def test_bio_within_platform_limit(self, smith):
        platform = get_platform("gumroad")
        profile = smith.generate_profile("gumroad")
        if platform.bio_max_length > 0:
            assert len(profile.bio) <= platform.bio_max_length, (
                f"Bio {len(profile.bio)} exceeds limit {platform.bio_max_length}"
            )

    def test_username_within_platform_limit(self, smith):
        platform = get_platform("gumroad")
        profile = smith.generate_profile("gumroad")
        assert len(profile.username) <= platform.username_max_length

    def test_tagline_within_platform_limit(self, smith):
        platform = get_platform("gumroad")
        profile = smith.generate_profile("gumroad")
        if platform.tagline_max_length > 0 and profile.tagline:
            assert len(profile.tagline) <= platform.tagline_max_length

    def test_description_within_platform_limit(self, smith):
        platform = get_platform("gumroad")
        profile = smith.generate_profile("gumroad")
        if platform.description_max_length > 0 and profile.description:
            assert len(profile.description) <= platform.description_max_length

    def test_social_links_within_limit(self, smith):
        platform = get_platform("gumroad")
        profile = smith.generate_profile("gumroad")
        if platform.max_links > 0:
            assert len(profile.social_links) <= platform.max_links

    def test_banner_cleared_when_unsupported(self, smith):
        """Platforms that don't support banners should have empty banner_path."""
        # Find a platform that doesn't support banners
        platform = get_platform("gumroad")
        profile = smith.generate_profile("gumroad")
        if not platform.allows_banner:
            assert profile.banner_path == ""

    def test_seo_keywords_populated(self, smith):
        profile = smith.generate_profile("gumroad")
        assert isinstance(profile.seo_keywords, list)
        assert len(profile.seo_keywords) >= 1

    def test_multiple_calls_produce_varied_content(self, smith):
        """ProfileSmith should produce some variation across calls."""
        profiles = [smith.generate_profile("gumroad") for _ in range(5)]
        bios = {p.bio for p in profiles}
        # With randomized templates, we should get at least 2 unique bios
        # (with very high probability over 5 draws from a pool of 8+)
        assert len(bios) >= 2, "Expected varied bios across multiple generations"

    def test_generate_batch(self, smith):
        results = smith.generate_batch(["gumroad", "etsy"])
        assert isinstance(results, dict)
        assert "gumroad" in results
        assert "etsy" in results
        assert all(isinstance(v, ProfileContent) for v in results.values())

    def test_generate_batch_skips_invalid(self, smith):
        results = smith.generate_batch(["gumroad", "fake_xyz_123"])
        assert "gumroad" in results
        assert "fake_xyz_123" not in results

    def test_display_name_from_brand(self, smith):
        profile = smith.generate_profile("gumroad")
        assert profile.display_name, "display_name should be populated from brand"

    def test_username_is_alphanumeric(self, smith):
        profile = smith.generate_profile("gumroad")
        # Usernames should only contain alphanumeric + underscores
        assert all(c.isalnum() or c == "_" for c in profile.username), (
            f"Username '{profile.username}' has invalid characters"
        )
