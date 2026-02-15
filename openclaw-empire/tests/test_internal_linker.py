"""
Tests for the Internal Linker module.

Tests link suggestion, link injection into HTML, site-specific link
graphs, cluster-based linking, orphan detection, and pillar identification.
All WordPress API calls are mocked.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.internal_linker import (
        InternalLinker,
        LinkGraph,
        LinkOpportunity,
        LinkReport,
        PostNode,
        get_linker,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="internal_linker module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def linker_dir(tmp_path):
    """Isolated data directory for linker state."""
    d = tmp_path / "linker"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def linker(linker_dir):
    """Create InternalLinker with temp data dir."""
    site_configs = {
        "testsite1": {
            "domain": "testsite1.com",
            "wp_user": "admin",
            "wp_app_password": "xxxx",
        },
        "testsite2": {
            "domain": "testsite2.com",
            "wp_user": "admin",
            "wp_app_password": "xxxx",
        },
    }
    linker_obj = InternalLinker(data_dir=linker_dir)
    # Inject site configs directly to bypass file-based loading
    linker_obj._site_configs = site_configs
    return linker_obj


@pytest.fixture
def sample_graph():
    """Pre-built LinkGraph for testing."""
    nodes = {}
    for i in range(1, 6):
        node = PostNode(
            post_id=i,
            title=f"Post {i}",
            url=f"https://testsite1.com/post-{i}/",
            slug=f"post-{i}",
            categories=["witchcraft"] if i % 2 else ["crystals"],
            keywords=[f"keyword-{i}", "magic"],
            internal_links_out=[],
            internal_links_in=[],
        )
        nodes[i] = node
    # Create some link relationships
    nodes[1].internal_links_out = [2, 3]
    nodes[2].internal_links_in = [1]
    nodes[3].internal_links_in = [1]
    nodes[2].internal_links_out = [4]
    nodes[4].internal_links_in = [2]
    # Post 5 is an orphan (no incoming links)
    # Post 1 also has no incoming links
    orphan_ids = [pid for pid, n in nodes.items() if not n.internal_links_in]
    return LinkGraph(site_id="testsite1", nodes=nodes, orphan_pages=orphan_ids)


# ===================================================================
# PostNode Tests
# ===================================================================

class TestPostNode:
    """Test PostNode dataclass."""

    def test_create_post_node(self):
        node = PostNode(
            post_id=1,
            title="Test Post",
            url="https://test.com/post-1/",
            slug="post-1",
            categories=["witchcraft"],
            keywords=["moon", "ritual"],
            internal_links_out=[2, 3],
            internal_links_in=[4],
        )
        assert node.post_id == 1
        assert node.incoming_count == 1
        assert node.outgoing_count == 2
        assert node.total_links == 3

    def test_orphan_detection(self):
        orphan = PostNode(
            post_id=99,
            title="Lonely Post",
            url="https://test.com/lonely/",
            slug="lonely",
            categories=[],
            keywords=[],
            internal_links_out=[],
            internal_links_in=[],
        )
        assert orphan.is_orphan is True

    def test_non_orphan(self):
        node = PostNode(
            post_id=1,
            title="Connected",
            url="https://test.com/connected/",
            slug="connected",
            categories=[],
            keywords=[],
            internal_links_out=[2],
            internal_links_in=[3],
        )
        assert node.is_orphan is False

    def test_to_dict_from_dict(self):
        node = PostNode(
            post_id=1, title="Test", url="https://test.com/",
            slug="test", categories=["cat"], keywords=["kw"],
            internal_links_out=[2], internal_links_in=[3],
        )
        d = node.to_dict()
        restored = PostNode.from_dict(d)
        assert restored.post_id == node.post_id
        assert restored.title == node.title


# ===================================================================
# LinkGraph Tests
# ===================================================================

class TestLinkGraph:
    """Test LinkGraph container."""

    def test_graph_post_count(self, sample_graph):
        assert sample_graph.post_count == 5

    def test_graph_edge_count(self, sample_graph):
        assert sample_graph.edge_count >= 0

    def test_graph_to_dict(self, sample_graph):
        d = sample_graph.to_dict()
        assert "site_id" in d
        assert "nodes" in d

    def test_graph_from_dict(self, sample_graph):
        d = sample_graph.to_dict()
        restored = LinkGraph.from_dict(d)
        assert restored.site_id == "testsite1"
        assert restored.post_count == 5


# ===================================================================
# LinkOpportunity Tests
# ===================================================================

class TestLinkOpportunity:
    """Test link suggestion dataclass."""

    def test_create_opportunity(self):
        opp = LinkOpportunity(
            source_post_id=1,
            target_post_id=2,
            anchor_text="moon water guide",
            context_sentence="Learn more about the moon water guide for beginners.",
            relevance_score=0.85,
            link_type="contextual",
        )
        assert opp.relevance_score == 0.85

    def test_opportunity_serialization(self):
        opp = LinkOpportunity(
            source_post_id=1, target_post_id=2,
            anchor_text="test",
            context_sentence="This is a test context.",
            relevance_score=0.5,
            link_type="related",
        )
        d = opp.to_dict()
        restored = LinkOpportunity.from_dict(d)
        assert restored.source_post_id == 1


# ===================================================================
# Build Graph Tests
# ===================================================================

class TestBuildGraph:
    """Test graph building from WordPress."""

    @pytest.mark.asyncio
    async def test_build_graph(self, linker):
        mock_wp = MagicMock()
        mock_wp.get_categories = AsyncMock(return_value=[
            {"id": 1, "name": "Witchcraft"},
        ])
        mock_wp.get_tags = AsyncMock(return_value=[])
        mock_wp.list_posts = AsyncMock(return_value=[
            {"id": 1, "title": {"rendered": "Post 1"}, "slug": "post-1",
             "link": "https://testsite1.com/post-1/", "content": {"rendered": "<p>Hello <a href='/post-2/'>link</a></p>"},
             "categories": [1], "tags": [], "date": "2026-01-01T00:00:00"},
            {"id": 2, "title": {"rendered": "Post 2"}, "slug": "post-2",
             "link": "https://testsite1.com/post-2/", "content": {"rendered": "<p>World</p>"},
             "categories": [1], "tags": [], "date": "2026-01-02T00:00:00"},
        ])
        with patch.object(linker, "_get_wp_client", return_value=mock_wp):
            graph = await linker.build_graph("testsite1", max_posts=10)
            assert isinstance(graph, LinkGraph)
            assert graph.post_count >= 1


# ===================================================================
# Link Health Tests
# ===================================================================

class TestLinkHealth:
    """Test link health reporting."""

    @pytest.mark.asyncio
    async def test_link_health(self, linker, sample_graph):
        # Pre-load the graph
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            report = await linker.link_health("testsite1")
            assert isinstance(report, LinkReport)

    def test_link_health_sync(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            report = linker.link_health_sync("testsite1")
            assert isinstance(report, LinkReport)


# ===================================================================
# Suggest Links Tests
# ===================================================================

class TestSuggestLinks:
    """Test link suggestion algorithms."""

    @pytest.mark.asyncio
    async def test_suggest_links(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            suggestions = await linker.suggest_links("testsite1", post_id=1)
            assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_suggest_links_for_content(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            html = "<p>This article is about moon water and crystal charging rituals.</p>"
            suggestions = await linker.suggest_links_for_content(
                site_id="testsite1",
                title="Moon Water Guide",
                content_html=html,
            )
            assert isinstance(suggestions, list)


# ===================================================================
# Inject Links Tests
# ===================================================================

class TestInjectLinks:
    """Test link injection into HTML content."""

    def test_inject_links(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        html = "<p>This is about moon water rituals and crystal magic.</p>"
        opportunities = [
            LinkOpportunity(
                source_post_id=1, target_post_id=2,
                anchor_text="crystal magic",
                context_sentence="This is about moon water rituals and crystal magic.",
                relevance_score=0.9,
                link_type="contextual",
                target_url="https://testsite1.com/post-2/",
            ),
        ]
        result = linker.inject_links(html, opportunities)
        assert isinstance(result, str)
        # Should contain original text
        assert "moon water" in result


# ===================================================================
# Orphan and Pillar Detection Tests
# ===================================================================

class TestOrphanAndPillar:
    """Test orphan detection and pillar identification."""

    @pytest.mark.asyncio
    async def test_find_orphans(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            orphans = await linker.find_orphans("testsite1")
            assert isinstance(orphans, list)
            assert any(n.post_id == 5 for n in orphans)  # Post 5 has no incoming links

    @pytest.mark.asyncio
    async def test_identify_pillars(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            pillars = await linker.identify_pillars("testsite1")
            assert isinstance(pillars, list)

    @pytest.mark.asyncio
    async def test_cluster_posts(self, linker, sample_graph):
        linker._graphs = {"testsite1": sample_graph}
        with patch.object(linker, "_ensure_graph", new_callable=AsyncMock, return_value=sample_graph):
            clusters = await linker.cluster_posts("testsite1")
            assert isinstance(clusters, dict)


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_linker_returns_instance(self, tmp_path):
        with patch.object(InternalLinker, "_load_site_configs", return_value={}):
            lnk = get_linker()
            assert isinstance(lnk, InternalLinker)
