"""Tests for content writer and voice system."""

import pytest
from unittest.mock import patch, MagicMock

from content.models import Article, Review, Listing, Post, ContentType, Difficulty
from content.voice import get_voice_prompt, get_profile_for_content_type, load_voice_profiles


# ── Voice Tests ──────────────────────────────────────────────────────────────

class TestVoice:
    def test_load_voice_profiles(self):
        profiles = load_voice_profiles()
        assert "profiles" in profiles
        assert "banned_phrases" in profiles
        assert "maker_mentor" in profiles["profiles"]
        assert "gear_reviewer" in profiles["profiles"]
        assert "community_voice" in profiles["profiles"]

    def test_get_voice_prompt(self):
        prompt = get_voice_prompt("maker_mentor")
        assert "Maker Mentor" in prompt
        assert "dive into" in prompt.lower()  # Should be in banned list
        assert "Writing rules:" in prompt

    def test_get_voice_prompt_invalid(self):
        with pytest.raises(ValueError, match="Unknown voice profile"):
            get_voice_prompt("nonexistent_profile")

    def test_banned_phrases_included(self):
        prompt = get_voice_prompt("maker_mentor")
        assert "NEVER use these phrases" in prompt
        assert '"In this article, we will"' in prompt

    def test_vocabulary_preferences(self):
        prompt = get_voice_prompt("maker_mentor")
        assert "dial in" in prompt

    def test_profile_for_content_type(self):
        assert get_profile_for_content_type("article") == "maker_mentor"
        assert get_profile_for_content_type("review") == "gear_reviewer"
        assert get_profile_for_content_type("post") == "community_voice"
        assert get_profile_for_content_type("listing") == "maker_mentor"
        assert get_profile_for_content_type("unknown") == "maker_mentor"


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestModels:
    def test_article_creation(self):
        article = Article(title="Test Article")
        assert article.content_type == ContentType.ARTICLE
        assert article.difficulty == Difficulty.BEGINNER
        assert article.word_count == 0

    def test_article_slug(self):
        article = Article(title="How to Print in Vase Mode!")
        slug = article.to_slug()
        assert slug == "how-to-print-in-vase-mode"
        assert article.slug == slug

    def test_article_markdown(self):
        from content.models import Section
        article = Article(
            title="Test",
            intro="Intro text",
            sections=[Section(heading="Step 1", body="Do this")],
            conclusion="The end",
        )
        md = article.full_markdown()
        assert "# Test" in md
        assert "Intro text" in md
        assert "## Step 1" in md
        assert "Do this" in md
        assert "## Wrapping Up" in md
        assert "The end" in md

    def test_article_word_count(self):
        article = Article(title="Test", intro="One two three")
        count = article.compute_word_count()
        assert count > 0

    def test_review_creation(self):
        review = Review(title="Printer Review", product_name="Ender 3")
        assert review.content_type == ContentType.REVIEW
        assert review.product_name == "Ender 3"
        assert review.rating == 0.0

    def test_review_markdown(self):
        review = Review(
            title="Test Review",
            overview="Great printer",
            pros=["Fast", "Cheap"],
            cons=["Loud"],
            best_for="Beginners",
            skip_if="Speed matters",
        )
        md = review.full_markdown()
        assert "Great printer" in md
        assert "Fast" in md
        assert "Loud" in md
        assert "Beginners" in md

    def test_listing_creation(self):
        listing = Listing(title="Cool Model", product_name="Vase")
        assert listing.content_type == ContentType.LISTING
        assert "STL" in listing.file_formats

    def test_post_creation(self):
        post = Post(title="Quick Tip", body="Use 210C for PLA")
        assert post.content_type == ContentType.POST
        md = post.full_markdown()
        assert "Quick Tip" in md
        assert "210C" in md


# ── Writer Tests (Mocked API) ───────────────────────────────────────────────

class TestWriter:
    @patch("content.writer.anthropic.Anthropic")
    def test_writer_classify(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="beginner")]
        mock_response.usage = MagicMock(
            input_tokens=50, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        from content.writer import ContentWriter
        writer = ContentWriter(api_key="test")
        result = writer.classify("vase mode", ["beginner", "intermediate", "advanced"])
        assert result == "beginner"
        assert writer.stats.calls == 1

    @patch("content.writer.anthropic.Anthropic")
    def test_writer_generate_tags(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="3d printing, vase mode, spiral, pla, single wall")]
        mock_response.usage = MagicMock(
            input_tokens=50, output_tokens=20,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        from content.writer import ContentWriter
        writer = ContentWriter(api_key="test")
        tags = writer.generate_tags("Vase Mode Guide", "article")
        assert isinstance(tags, list)
        assert len(tags) <= 10
        assert "3d printing" in tags

    @patch("content.writer.anthropic.Anthropic")
    def test_writer_cost_tracking(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        from content.writer import ContentWriter
        writer = ContentWriter(api_key="test")
        writer.classify("test", ["a", "b"])

        summary = writer.get_cost_summary()
        assert summary["total_calls"] == 1
        assert summary["total_input_tokens"] == 100
        assert summary["total_output_tokens"] == 50
        assert summary["total_cost_usd"] > 0

    @patch("content.writer.anthropic.Anthropic")
    def test_cache_control_for_long_prompts(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        from content.writer import ContentWriter
        writer = ContentWriter(api_key="test")

        # Short prompt — no caching
        system = writer._build_system("short prompt")
        assert isinstance(system, str)

        # Long prompt — should get cache_control
        long_prompt = "x" * 3000
        system = writer._build_system(long_prompt)
        assert isinstance(system, list)
        assert system[0]["cache_control"]["type"] == "ephemeral"
