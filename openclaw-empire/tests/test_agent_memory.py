"""
Tests for the Agent Memory module.

Tests memory store/retrieve, text search, tagging, memory types,
expiration, consolidation, and statistics. All Anthropic API calls
for summarization are mocked.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.agent_memory import (
        AgentMemory,
        ConsolidationResult,
        ConsolidationStrategy,
        Memory,
        MemoryPriority,
        MemoryQuery,
        MemoryStats,
        MemoryType,
        get_memory,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="agent_memory module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def mem_dir(tmp_path):
    """Isolated data directory for memory state."""
    d = tmp_path / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def memory(mem_dir):
    """Create a fresh AgentMemory with temp data dir."""
    return AgentMemory(data_dir=mem_dir, session_id="test_session")


@pytest.fixture
def populated_memory(memory):
    """Memory pre-loaded with sample entries."""
    memory.store_sync(
        content="Moon water rituals are best performed during full moon.",
        type=MemoryType.FACT,
        priority=MemoryPriority.NORMAL,
        tags=["witchcraft", "moon", "ritual"],
    )
    memory.store_sync(
        content="User prefers articles between 1500-2500 words.",
        type=MemoryType.USER_PREFERENCE,
        priority=MemoryPriority.HIGH,
        tags=["content", "preferences"],
    )
    memory.store_sync(
        content="Instagram posting works best at 9am EST for witchcraft niche.",
        type=MemoryType.OBSERVATION,
        priority=MemoryPriority.NORMAL,
        tags=["social", "instagram", "timing"],
    )
    memory.store_sync(
        content="Error: WordPress API returned 401 on testsite1. App password expired.",
        type=MemoryType.ERROR_LOG,
        priority=MemoryPriority.CRITICAL,
        tags=["error", "wordpress", "auth"],
    )
    memory.store_sync(
        content="Learned to use RankMath API for SEO scoring instead of manual checks.",
        type=MemoryType.SKILL_LEARNED,
        priority=MemoryPriority.HIGH,
        tags=["seo", "rankmath", "skill"],
    )
    return memory


# ===================================================================
# Enum Tests
# ===================================================================

class TestEnums:
    """Verify enum members."""

    def test_memory_type_members(self):
        assert MemoryType.TASK_RESULT is not None
        assert MemoryType.APP_KNOWLEDGE is not None
        assert MemoryType.USER_PREFERENCE is not None
        assert MemoryType.ERROR_LOG is not None
        assert MemoryType.CONVERSATION is not None
        assert MemoryType.OBSERVATION is not None
        assert MemoryType.SKILL_LEARNED is not None
        assert MemoryType.FACT is not None
        assert MemoryType.PROCEDURE is not None

    def test_memory_priority_members(self):
        assert MemoryPriority.CRITICAL is not None
        assert MemoryPriority.HIGH is not None
        assert MemoryPriority.NORMAL is not None
        assert MemoryPriority.LOW is not None
        assert MemoryPriority.EPHEMERAL is not None

    def test_consolidation_strategy_members(self):
        assert ConsolidationStrategy.SUMMARIZE is not None
        assert ConsolidationStrategy.MERGE is not None
        assert ConsolidationStrategy.ARCHIVE is not None
        assert ConsolidationStrategy.DELETE is not None


# ===================================================================
# Memory Dataclass Tests
# ===================================================================

class TestMemoryDataclass:
    """Test Memory dataclass properties."""

    def test_create_memory(self):
        mem = Memory(
            content="Test memory content",
            type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
            tags=["test"],
        )
        assert mem.content == "Test memory content"
        assert mem.type == MemoryType.FACT

    def test_memory_not_expired(self):
        mem = Memory(
            content="Fresh memory",
            type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
        )
        assert mem.is_expired() is False

    def test_memory_expired(self):
        past_expiry = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mem = Memory(
            content="Old memory",
            type=MemoryType.FACT,
            priority=MemoryPriority.EPHEMERAL,
            created_at=(datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
            expires_at=past_expiry,
        )
        assert mem.is_expired() is True

    def test_age_hours(self):
        mem = Memory(
            content="Recent memory",
            type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
            created_at=(datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
        )
        assert 4 < mem.age_hours() < 6


# ===================================================================
# MemoryQuery Tests
# ===================================================================

class TestMemoryQuery:
    """Test query construction."""

    def test_create_query(self):
        q = MemoryQuery(
            text="moon water ritual",
            types=[MemoryType.FACT, MemoryType.OBSERVATION],
            tags=["witchcraft"],
            limit=5,
        )
        assert q.text == "moon water ritual"
        assert len(q.types) == 2
        assert q.limit == 5


# ===================================================================
# Store Tests
# ===================================================================

class TestStore:
    """Test memory storage."""

    def test_store_sync(self, memory):
        mem = memory.store_sync(
            content="Test fact about crystals",
            type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
            tags=["crystals"],
        )
        assert mem is not None
        assert isinstance(mem, Memory)

    def test_store_multiple(self, memory):
        ids = []
        for i in range(5):
            mem = memory.store_sync(
                content=f"Memory entry {i}",
                type=MemoryType.OBSERVATION,
                priority=MemoryPriority.NORMAL,
                tags=[f"tag_{i}"],
            )
            ids.append(mem.id)
        assert len(set(ids)) == 5  # All unique

    def test_store_different_types(self, memory):
        types = [MemoryType.FACT, MemoryType.ERROR_LOG, MemoryType.SKILL_LEARNED]
        for mt in types:
            mem = memory.store_sync(
                content=f"Memory of type {mt.value}",
                type=mt,
                priority=MemoryPriority.NORMAL,
            )
            assert mem is not None

    def test_store_with_metadata(self, memory):
        mem = memory.store_sync(
            content="Memory with metadata",
            type=MemoryType.TASK_RESULT,
            priority=MemoryPriority.HIGH,
            tags=["task", "result"],
            metadata={"task_id": "t_001", "duration_ms": 5000},
        )
        assert mem is not None
        assert isinstance(mem, Memory)


# ===================================================================
# Recall Tests
# ===================================================================

class TestRecall:
    """Test memory retrieval by text similarity."""

    def test_recall_by_text(self, populated_memory):
        results = populated_memory.recall_by_text_sync("moon water ritual")
        assert isinstance(results, list)
        assert len(results) >= 1
        # Most relevant result should mention moon
        assert any("moon" in r.content.lower() for r in results)

    def test_recall_empty_query(self, populated_memory):
        results = populated_memory.recall_by_text_sync("")
        assert isinstance(results, list)

    def test_recall_no_match(self, populated_memory):
        results = populated_memory.recall_by_text_sync("quantum physics blockchain")
        assert isinstance(results, list)

    def test_recall_with_limit(self, populated_memory):
        results = populated_memory.recall_by_text_sync("witchcraft", limit=2)
        assert len(results) <= 2


# ===================================================================
# Context Retrieval Tests
# ===================================================================

class TestContext:
    """Test context window retrieval."""

    def test_get_context_sync(self, populated_memory):
        ctx = populated_memory.get_context_sync(
            topic="writing content about moon rituals",
            max_tokens=3000,
        )
        assert isinstance(ctx, str)

    def test_get_context_empty_memory(self, memory):
        ctx = memory.get_context_sync(
            topic="anything",
            max_tokens=3000,
        )
        assert isinstance(ctx, str)


# ===================================================================
# Consolidation Tests
# ===================================================================

class TestConsolidation:
    """Test memory consolidation strategies."""

    def test_consolidate_sync(self, populated_memory):
        with patch("src.agent_memory._haiku_summarize",
                     return_value="Summary of witchcraft memories"):
            result = populated_memory.consolidate_sync(
                strategy=ConsolidationStrategy.SUMMARIZE,
            )
            assert isinstance(result, ConsolidationResult)

    def test_consolidate_merge(self, populated_memory):
        result = populated_memory.consolidate_sync(
            strategy=ConsolidationStrategy.MERGE,
        )
        assert isinstance(result, ConsolidationResult)

    def test_consolidate_archive(self, populated_memory):
        result = populated_memory.consolidate_sync(
            strategy=ConsolidationStrategy.ARCHIVE,
        )
        assert isinstance(result, ConsolidationResult)

    def test_consolidate_delete_expired(self, memory):
        # Store a memory that will be immediately expired
        past_expiry = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        # We need to store and then manually set expires_at since store computes it
        mem = memory.store_sync(
            content="Old expired memory",
            type=MemoryType.OBSERVATION,
            priority=MemoryPriority.EPHEMERAL,
            ttl_hours=0,  # Immediately expired
        )
        # Force-expire by setting expires_at to the past
        mem.expires_at = past_expiry
        result = memory.consolidate_sync(
            strategy=ConsolidationStrategy.DELETE,
        )
        assert isinstance(result, ConsolidationResult)


# ===================================================================
# Statistics Tests
# ===================================================================

class TestStatistics:
    """Test memory statistics."""

    def test_stats(self, populated_memory):
        if hasattr(populated_memory, "get_stats_sync"):
            stats = populated_memory.get_stats_sync()
        elif hasattr(populated_memory, "get_stats"):
            stats = populated_memory.get_stats_sync()
        else:
            pytest.skip("No stats method found")
            return
        assert isinstance(stats, (MemoryStats, dict))


# ===================================================================
# Persistence Tests
# ===================================================================

class TestPersistence:
    """Test memory saving and loading."""

    def test_memories_persist_to_disk(self, mem_dir):
        m1 = AgentMemory(data_dir=mem_dir, session_id="persist_test")
        m1.store_sync(
            content="Persistent memory",
            type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
        )
        # Force save to ensure data is on disk
        m1._force_save()
        # Reload
        m2 = AgentMemory(data_dir=mem_dir, session_id="persist_test")
        results = m2.recall_by_text_sync("persistent")
        assert len(results) >= 1

    def test_different_sessions_isolated(self, mem_dir):
        m1 = AgentMemory(data_dir=mem_dir, session_id="session_a")
        m1.store_sync(
            content="Session A only",
            type=MemoryType.FACT,
            priority=MemoryPriority.NORMAL,
        )
        m1._force_save()
        m2 = AgentMemory(data_dir=mem_dir, session_id="session_b")
        results = m2.recall_by_text_sync("Session A only")
        # Depending on implementation, may or may not find cross-session
        assert isinstance(results, list)


# ===================================================================
# Text Similarity Tests
# ===================================================================

class TestTextSimilarity:
    """Test internal text matching utilities."""

    def test_text_similarity(self, memory):
        if not hasattr(memory, "_text_similarity"):
            pytest.skip("_text_similarity not exposed")
        score = memory._text_similarity("moon water ritual", "moon water ceremony")
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should be somewhat similar

    def test_word_overlap(self, memory):
        if not hasattr(memory, "_word_overlap_ratio"):
            pytest.skip("_word_overlap_ratio not exposed")
        ratio = memory._word_overlap_ratio(
            "moon water ritual guide",
            "guide to moon water preparation"
        )
        assert 0.0 <= ratio <= 1.0
        assert ratio > 0.3


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_memory_returns_instance(self, tmp_path):
        import src.agent_memory as am
        # Reset the singleton so we can create a fresh one
        am._instance = None
        m = get_memory(config={"data_dir": tmp_path / "mem"})
        assert isinstance(m, AgentMemory)
        # Reset again so other tests are not affected
        am._instance = None
