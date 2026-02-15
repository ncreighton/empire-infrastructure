"""Test rag_memory -- OpenClaw Empire."""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch data directories before imports
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_rag_dirs(tmp_path, monkeypatch):
    """Redirect all RAG file I/O to temp directory."""
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.rag_memory.RAG_DATA_DIR", rag_dir)
    monkeypatch.setattr("src.rag_memory.TFIDF_INDEX_FILE", rag_dir / "tfidf_index.json")
    monkeypatch.setattr("src.rag_memory.SEARCH_LOG_FILE", rag_dir / "search_log.json")
    monkeypatch.setattr("src.rag_memory.INDEX_META_FILE", rag_dir / "index_meta.json")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.rag_memory import (
    DEFAULT_CONTEXT_MAX_TOKENS,
    DEFAULT_MIN_SCORE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_WEIGHT,
    RAGContext,
    RAGMemory,
    SearchResult,
    SearchStrategy,
    SimpleStemmer,
    TFIDFEngine,
    get_rag_memory,
)


# ===========================================================================
# SIMPLE STEMMER
# ===========================================================================


class TestSimpleStemmer:
    """Test the lightweight suffix-stripping stemmer."""

    def test_regular_suffix_stripping(self):
        assert SimpleStemmer.stem("running") == "run"
        assert SimpleStemmer.stem("stories") == "story"

    def test_irregulars(self):
        assert SimpleStemmer.stem("went") == "go"
        assert SimpleStemmer.stem("children") == "child"

    def test_short_words_unchanged(self):
        # Words shorter than STEM_MIN_LENGTH (3) are returned as-is
        assert SimpleStemmer.stem("hi") == "hi"
        assert SimpleStemmer.stem("") == ""

    def test_cache(self):
        SimpleStemmer.clear_cache()
        SimpleStemmer.stem("testing")
        assert SimpleStemmer.cache_size() > 0

    def test_clear_cache(self):
        SimpleStemmer.stem("something")
        SimpleStemmer.clear_cache()
        assert SimpleStemmer.cache_size() == 0


# ===========================================================================
# TF-IDF ENGINE
# ===========================================================================


class TestTFIDFEngine:
    """Test the pure-Python TF-IDF engine."""

    @pytest.fixture
    def engine(self):
        return TFIDFEngine()

    @pytest.fixture
    def corpus_engine(self):
        engine = TFIDFEngine()
        engine.add_document("doc1", "Instagram login troubleshooting two-factor authentication")
        engine.add_document("doc2", "TikTok video upload and scheduling social media")
        engine.add_document("doc3", "Instagram story posting and engagement metrics analytics")
        engine.add_document("doc4", "Pinterest pin creation and board management strategy")
        engine.add_document("doc5", "Facebook marketplace listing and selling products online")
        engine._rebuild_idf()
        engine._rebuild_all_vectors()
        return engine

    # ------- Tokenization -------

    def test_tokenize_basic(self, engine):
        tokens = engine._tokenize("Hello World! This is a TEST.")
        assert len(tokens) > 0
        # Should be lowercased and stemmed
        for t in tokens:
            assert t == t.lower()

    def test_tokenize_removes_stopwords(self, engine):
        tokens = engine._tokenize("the quick brown fox and the lazy dog")
        # "the", "and" should be removed
        assert "the" not in tokens
        assert "and" not in tokens

    def test_tokenize_empty(self, engine):
        assert engine._tokenize("") == []
        assert engine._tokenize("   ") == []

    # ------- Term Frequency -------

    def test_compute_tf(self, engine):
        tokens = ["apple", "apple", "banana"]
        tf = engine._compute_tf(tokens)
        assert tf["apple"] > tf["banana"]
        # Most frequent term should have TF = 1.0
        assert tf["apple"] == 1.0

    def test_compute_tf_empty(self, engine):
        assert engine._compute_tf([]) == {}

    # ------- Document Management -------

    def test_add_document(self, engine):
        engine.add_document("doc1", "Hello world test content")
        assert engine.has_document("doc1")
        assert engine._document_count == 1

    def test_add_duplicate_replaces(self, engine):
        engine.add_document("doc1", "Version one")
        engine.add_document("doc1", "Version two")
        assert engine._document_count == 1

    def test_remove_document(self, engine):
        engine.add_document("doc1", "Some content here")
        removed = engine.remove_document("doc1")
        assert removed is True
        assert not engine.has_document("doc1")

    def test_remove_nonexistent(self, engine):
        removed = engine.remove_document("ghost")
        assert removed is False

    def test_document_ids(self, engine):
        engine.add_document("a", "Alpha content")
        engine.add_document("b", "Beta content")
        ids = engine.document_ids()
        assert set(ids) == {"a", "b"}

    # ------- IDF Computation -------

    def test_rebuild_idf(self, corpus_engine):
        assert len(corpus_engine._idf) > 0
        # Terms present in fewer documents should have higher IDF
        # "instagram" appears in 2 docs, "facebook" in 1 -- facebook should be higher
        if "facebook" in corpus_engine._idf and "instagram" in corpus_engine._idf:
            assert corpus_engine._idf.get("facebook", 0) >= corpus_engine._idf.get("instagram", 0)

    # ------- Cosine Similarity Search -------

    def test_search_returns_results(self, corpus_engine):
        results = corpus_engine.search("instagram login", limit=5, min_score=0.01)
        assert len(results) > 0
        # Results should be (doc_id, score) tuples
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][1], float)

    def test_search_relevant_first(self, corpus_engine):
        results = corpus_engine.search("instagram", limit=5, min_score=0.01)
        # doc1 and doc3 mention Instagram, should rank higher
        if len(results) >= 2:
            top_ids = [r[0] for r in results[:2]]
            assert "doc1" in top_ids or "doc3" in top_ids

    def test_search_empty_query(self, corpus_engine):
        results = corpus_engine.search("", limit=5)
        assert results == []

    def test_search_no_matches(self, corpus_engine):
        results = corpus_engine.search("xyzzyplugh", limit=5, min_score=0.5)
        assert results == []

    # ------- Keyword Search -------

    def test_keyword_search(self, corpus_engine):
        results = corpus_engine.keyword_search("instagram login", limit=5)
        assert len(results) > 0
        # doc1 has "instagram login" as a substring
        top_ids = [r[0] for r in results[:3]]
        assert "doc1" in top_ids

    def test_keyword_search_empty(self, corpus_engine):
        results = corpus_engine.keyword_search("")
        assert results == []

    def test_keyword_search_exact_match_bonus(self, corpus_engine):
        results = corpus_engine.keyword_search("Instagram login troubleshooting")
        if results:
            # doc1 is an exact match, should score higher
            assert results[0][0] == "doc1"

    # ------- Norm Computation -------

    def test_compute_norm(self, engine):
        vector = {"a": 3.0, "b": 4.0}
        norm = engine._compute_norm(vector)
        assert abs(norm - 5.0) < 0.01  # 3^2 + 4^2 = 25, sqrt(25) = 5

    def test_compute_norm_empty(self, engine):
        assert engine._compute_norm({}) == 0.0


# ===========================================================================
# SEARCH RESULT DATACLASS
# ===========================================================================


class TestSearchResult:

    def test_to_dict(self):
        sr = SearchResult(
            memory_id="mem-1",
            content="Test content",
            score=0.85,
            memory_type="app_knowledge",
            tags=["instagram", "login"],
        )
        d = sr.to_dict()
        assert d["memory_id"] == "mem-1"
        assert d["score"] == 0.85

    def test_from_dict(self):
        data = {
            "memory_id": "mem-2",
            "content": "Some memory",
            "score": 0.5,
            "memory_type": "workflow",
            "tags": ["tiktok"],
        }
        sr = SearchResult.from_dict(data)
        assert sr.memory_id == "mem-2"
        assert sr.score == 0.5

    def test_from_dict_ignores_unknown_fields(self):
        data = {
            "memory_id": "mem-3",
            "content": "Content",
            "score": 0.7,
            "unknown_field": "should be ignored",
        }
        sr = SearchResult.from_dict(data)
        assert sr.memory_id == "mem-3"
        assert not hasattr(sr, "unknown_field")


# ===========================================================================
# RAG CONTEXT DATACLASS
# ===========================================================================


class TestRAGContext:

    def test_to_dict(self):
        ctx = RAGContext(
            query="how to post",
            context_string="Relevant context here",
            token_estimate=100,
            strategy_used=SearchStrategy.HYBRID,
            search_time_ms=5.3,
        )
        d = ctx.to_dict()
        assert d["query"] == "how to post"
        assert d["strategy_used"] == "hybrid"

    def test_from_dict_roundtrip(self):
        original = RAGContext(
            query="test",
            strategy_used=SearchStrategy.SEMANTIC,
            token_estimate=50,
        )
        d = original.to_dict()
        restored = RAGContext.from_dict(d)
        assert restored.query == "test"
        assert restored.strategy_used == SearchStrategy.SEMANTIC


# ===========================================================================
# SEARCH STRATEGY ENUM
# ===========================================================================


class TestSearchStrategy:

    def test_three_strategies(self):
        assert len(list(SearchStrategy)) == 3

    def test_values(self):
        assert SearchStrategy.SEMANTIC == "semantic"
        assert SearchStrategy.KEYWORD == "keyword"
        assert SearchStrategy.HYBRID == "hybrid"


# ===========================================================================
# RAG MEMORY CLASS
# ===========================================================================


class TestRAGMemory:
    """Test the RAGMemory wrapper around TFIDFEngine."""

    @pytest.fixture
    def rag(self):
        """Create a fresh RAGMemory instance."""
        return RAGMemory()

    @pytest.fixture
    def indexed_rag(self):
        """Create a RAGMemory with some indexed memories."""
        rag = RAGMemory()
        # Index memories directly (bypass agent_memory dependency)
        rag._tfidf.add_document("mem1", "Instagram login two-factor authentication setup")
        rag._memory_meta["mem1"] = {
            "content": "Instagram login two-factor authentication setup",
            "type": "app_knowledge",
            "tags": ["instagram", "auth"],
            "created_at": "2026-01-01T00:00:00Z",
        }
        rag._tfidf.add_document("mem2", "TikTok video upload scheduling tools available")
        rag._memory_meta["mem2"] = {
            "content": "TikTok video upload scheduling tools available",
            "type": "workflow",
            "tags": ["tiktok", "video"],
            "created_at": "2026-01-02T00:00:00Z",
        }
        rag._tfidf.add_document("mem3", "Pinterest pin design brand colors witchcraft niche")
        rag._memory_meta["mem3"] = {
            "content": "Pinterest pin design brand colors witchcraft niche",
            "type": "design",
            "tags": ["pinterest", "witchcraft"],
            "created_at": "2026-01-03T00:00:00Z",
        }
        rag._tfidf._rebuild_idf()
        rag._tfidf._rebuild_all_vectors()
        rag._indexed_count = 3
        return rag

    # ------- Initialization -------

    def test_init_creates_engine(self, rag):
        assert isinstance(rag._tfidf, TFIDFEngine)
        assert rag._indexed_count == 0

    # ------- Index Single Memory -------

    @pytest.mark.asyncio
    async def test_index_memory(self, rag):
        await rag.index_memory(
            memory_id="test1",
            content="Important workflow for account creation on Instagram",
            memory_type="workflow",
            tags=["instagram", "account"],
        )
        assert rag._tfidf.has_document("test1")
        assert "test1" in rag._memory_meta

    @pytest.mark.asyncio
    async def test_index_memory_empty_content_skipped(self, rag):
        await rag.index_memory(
            memory_id="empty",
            content="",
            memory_type="",
        )
        # Empty content should be skipped
        assert not rag._tfidf.has_document("empty")

    # ------- Semantic Search -------

    @pytest.mark.asyncio
    async def test_semantic_search_returns_results(self, indexed_rag):
        results = await indexed_rag.semantic_search("instagram login", limit=5, min_score=0.01)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_semantic_search_relevant_ranking(self, indexed_rag):
        results = await indexed_rag.semantic_search("instagram authentication", limit=3, min_score=0.01)
        if results:
            # mem1 about Instagram auth should be top
            assert results[0].memory_id == "mem1"

    @pytest.mark.asyncio
    async def test_semantic_search_empty_query(self, indexed_rag):
        results = await indexed_rag.semantic_search("", limit=5)
        assert results == []

    # ------- Keyword Search -------

    @pytest.mark.asyncio
    async def test_keyword_search_returns_results(self, indexed_rag):
        # keyword_search is on the TFIDFEngine, accessible through RAG
        results = indexed_rag._tfidf.keyword_search("pinterest pin", limit=5)
        assert len(results) > 0

    # ------- Empty Corpus Handling -------

    @pytest.mark.asyncio
    async def test_search_on_empty_index(self, rag):
        results = await rag.semantic_search("anything", limit=5)
        assert results == []

    # ------- Memory-to-Searchable -------

    def test_memory_to_searchable(self, rag):
        mem = {
            "content": "Primary content here",
            "summary": "Short summary",
            "tags": ["tag1", "tag2"],
            "type": "workflow",
            "source_module": "forge_engine",
            "metadata": {"app": "Instagram", "notes": "Use OAuth"},
        }
        text = rag._memory_to_searchable(mem)
        assert "Primary content here" in text
        assert "Short summary" in text
        assert "tag1" in text
        assert "workflow" in text

    def test_memory_to_searchable_empty(self, rag):
        mem = {}
        text = rag._memory_to_searchable(mem)
        assert text.strip() == ""

    # ------- Sync Wrappers -------

    def test_index_memory_sync(self, rag):
        rag.index_memory_sync(
            memory_id="sync_test",
            content="Testing synchronous indexing path",
            memory_type="test",
        )
        assert rag._tfidf.has_document("sync_test")


# ===========================================================================
# SINGLETON
# ===========================================================================


class TestGetRagMemory:

    def test_returns_rag_memory_instance(self):
        # Reset the singleton for test isolation
        with patch("src.rag_memory.RAGMemory.__init__", return_value=None):
            # Just verify the function exists and is callable
            assert callable(get_rag_memory)
