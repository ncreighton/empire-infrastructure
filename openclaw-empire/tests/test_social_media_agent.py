"""Test social_media_agent — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.social_media_agent import (
        SocialMediaAgent,
        ContentStrategy,
        EngagementAction,
        AnalyticsSnapshot,
        GrowthRecord,
        CompetitorProfile,
        DMConversation,
        SocialPlatform,
        ContentPillar,
        EngagementType,
        GrowthTactic,
        DMCategory,
        COMMENT_TEMPLATES,
        AUTO_RESPONSES,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="social_media_agent not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "social_agent"
    d.mkdir()
    return d


@pytest.fixture
def agent(data_dir):
    return SocialMediaAgent(data_dir=data_dir)


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_social_platform_values(self):
        assert SocialPlatform.INSTAGRAM.value == "instagram"
        assert SocialPlatform.YOUTUBE.value == "youtube"
        assert SocialPlatform.REDDIT.value == "reddit"
        assert SocialPlatform.THREADS.value == "threads"

    def test_content_pillar_values(self):
        assert ContentPillar.EDUCATIONAL.value == "educational"
        assert ContentPillar.ENTERTAINING.value == "entertaining"
        assert ContentPillar.BTS.value == "behind_the_scenes"
        assert ContentPillar.UGC.value == "user_generated_content"

    def test_engagement_type_values(self):
        assert EngagementType.LIKE.value == "like"
        assert EngagementType.COMMENT.value == "comment"
        assert EngagementType.STORY_REACT.value == "story_react"

    def test_growth_tactic_values(self):
        assert GrowthTactic.FOLLOW_UNFOLLOW.value == "follow_unfollow"
        assert GrowthTactic.HASHTAG_TARGETING.value == "hashtag_targeting"
        assert GrowthTactic.GIVEAWAY.value == "giveaway"

    def test_dm_category_values(self):
        assert DMCategory.INQUIRY.value == "inquiry"
        assert DMCategory.COLLABORATION.value == "collaboration"
        assert DMCategory.SPAM.value == "spam"


# ===================================================================
# Data Class Tests
# ===================================================================


class TestContentStrategy:
    def test_defaults(self):
        cs = ContentStrategy()
        assert cs.platform == SocialPlatform.INSTAGRAM
        assert cs.niche == ""
        assert cs.active is True

    def test_to_dict(self):
        cs = ContentStrategy(
            platform=SocialPlatform.TIKTOK,
            niche="witchcraft",
            pillars=[ContentPillar.EDUCATIONAL, ContentPillar.ENTERTAINING],
            growth_tactics=[GrowthTactic.HASHTAG_TARGETING],
        )
        d = cs.to_dict()
        assert d["platform"] == "tiktok"
        assert "educational" in d["pillars"]
        assert "hashtag_targeting" in d["growth_tactics"]


class TestEngagementAction:
    def test_defaults(self):
        ea = EngagementAction()
        assert ea.success is True
        assert ea.action_type == EngagementType.LIKE

    def test_to_dict(self):
        ea = EngagementAction(
            platform=SocialPlatform.TWITTER,
            action_type=EngagementType.COMMENT,
            comment_text="Great post!",
        )
        d = ea.to_dict()
        assert d["platform"] == "twitter"
        assert d["action_type"] == "comment"
        assert d["comment_text"] == "Great post!"


class TestAnalyticsSnapshot:
    def test_defaults(self):
        snap = AnalyticsSnapshot()
        assert snap.followers == 0
        assert snap.engagement_rate == 0.0

    def test_to_dict(self):
        snap = AnalyticsSnapshot(
            platform=SocialPlatform.INSTAGRAM,
            followers=5000,
            engagement_rate=3.5,
            reach=10000,
        )
        d = snap.to_dict()
        assert d["platform"] == "instagram"
        assert d["followers"] == 5000


class TestGrowthRecord:
    def test_defaults(self):
        gr = GrowthRecord()
        assert gr.new_followers == 0
        assert gr.lost_followers == 0

    def test_to_dict(self):
        gr = GrowthRecord(
            date="2026-02-15",
            platform=SocialPlatform.TIKTOK,
            followers=10000,
            new_followers=150,
        )
        d = gr.to_dict()
        assert d["platform"] == "tiktok"


class TestCompetitorProfile:
    def test_defaults(self):
        cp = CompetitorProfile()
        assert cp.username == ""
        assert cp.top_hashtags == []

    def test_to_dict(self):
        cp = CompetitorProfile(
            username="competitor1",
            platform=SocialPlatform.INSTAGRAM,
            followers=50000,
            avg_engagement=4.5,
        )
        d = cp.to_dict()
        assert d["username"] == "competitor1"


class TestDMConversation:
    def test_defaults(self):
        dm = DMConversation()
        assert dm.category == DMCategory.INQUIRY
        assert dm.auto_responded is False
        assert dm.escalated is False

    def test_to_dict(self):
        dm = DMConversation(
            user="follower123",
            category=DMCategory.COLLABORATION,
            messages=[{"from": "follower123", "text": "Want to collab?"}],
        )
        d = dm.to_dict()
        assert d["category"] == "collaboration"
        assert len(d["messages"]) == 1


# ===================================================================
# Comment Templates & Auto-Response Tests
# ===================================================================


class TestCommentTemplates:
    def test_witchcraft_templates_exist(self):
        assert "witchcraft" in COMMENT_TEMPLATES
        assert len(COMMENT_TEMPLATES["witchcraft"]) > 0

    def test_default_templates_exist(self):
        assert "default" in COMMENT_TEMPLATES

    def test_templates_have_placeholder(self):
        for niche, templates in COMMENT_TEMPLATES.items():
            for t in templates:
                assert "{specific}" in t, f"Missing {{specific}} in {niche} template"


class TestAutoResponses:
    def test_inquiry_response(self):
        assert DMCategory.INQUIRY in AUTO_RESPONSES
        assert "reach" in AUTO_RESPONSES[DMCategory.INQUIRY].lower() or "thanks" in AUTO_RESPONSES[DMCategory.INQUIRY].lower()

    def test_collaboration_response(self):
        assert DMCategory.COLLABORATION in AUTO_RESPONSES

    def test_support_response(self):
        assert DMCategory.SUPPORT in AUTO_RESPONSES


# ===================================================================
# SocialMediaAgent Tests
# ===================================================================


class TestSocialMediaAgentInit:
    def test_init_creates_data_dir(self, tmp_path):
        d = tmp_path / "new_social_agent"
        agent = SocialMediaAgent(data_dir=d)
        assert d.exists()

    def test_empty_state_on_init(self, agent):
        assert len(agent._strategies) == 0
        assert len(agent._actions) == 0
        assert len(agent._competitors) == 0


class TestSocialMediaAgentPersistence:
    def test_save_and_load(self, data_dir):
        agent1 = SocialMediaAgent(data_dir=data_dir)
        strategy = ContentStrategy(
            platform=SocialPlatform.INSTAGRAM,
            niche="witchcraft",
            pillars=[ContentPillar.EDUCATIONAL],
        )
        agent1._strategies[strategy.id] = strategy
        agent1._save_state()

        agent2 = SocialMediaAgent(data_dir=data_dir)
        assert len(agent2._strategies) == 1


class TestSocialMediaAgentAI:
    @pytest.mark.asyncio
    async def test_call_haiku_returns_empty_on_error(self, agent):
        with patch.dict("sys.modules", {"anthropic": None}):
            result = await agent._call_haiku("Generate a comment")
            assert result == ""

    @pytest.mark.asyncio
    async def test_call_sonnet_returns_empty_on_error(self, agent):
        with patch.dict("sys.modules", {"anthropic": None}):
            result = await agent._call_sonnet("Create strategy")
            assert result == ""


class TestSocialMediaAgentStrategy:
    @pytest.mark.asyncio
    async def test_generate_strategy_fallback(self, agent):
        """When AI call fails, strategy should use fallback defaults."""
        with patch.object(agent, "_call_sonnet", new_callable=AsyncMock, return_value=""):
            strategy = await agent.generate_strategy("instagram", "witchcraft")
            assert strategy.platform == SocialPlatform.INSTAGRAM
            assert strategy.niche == "witchcraft"
            assert len(strategy.pillars) > 0
            assert strategy.posting_frequency != ""
