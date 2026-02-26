"""Tests for VariationEngine — anti-repetition pool selector."""

import pytest
from videoforge.forge.variation_engine import VariationEngine


@pytest.fixture
def engine():
    e = VariationEngine(db_path=":memory:")
    yield e
    e.close()


class TestVariationEngine:
    def test_pick_returns_from_pool(self, engine):
        result = engine.pick("test_pool", ["a", "b", "c"])
        assert result in ["a", "b", "c"]

    def test_pick_empty_returns_empty(self, engine):
        assert engine.pick("test", []) == ""

    def test_pick_single_returns_that_item(self, engine):
        assert engine.pick("test", ["only"]) == "only"

    def test_pick_n_returns_n_unique(self, engine):
        results = engine.pick_n("test", ["a", "b", "c", "d", "e"], 3)
        assert len(results) == 3
        assert len(set(results)) == 3  # All unique

    def test_pick_n_capped_at_pool_size(self, engine):
        results = engine.pick_n("test", ["a", "b"], 5)
        assert len(results) == 2

    def test_pick_n_empty_pool(self, engine):
        results = engine.pick_n("test", [], 3)
        assert results == []

    def test_recency_bias_avoids_recent(self, engine):
        """After picking 'a' many times, other items should be favored."""
        pool = ["a", "b", "c", "d", "e"]
        # Pick 'a' 20 times by forcing it through single-item picks
        for _ in range(20):
            engine._log_selection("bias_test", "a")

        # Now pick 100 times and count 'a' occurrences
        results = [engine.pick("bias_test", pool) for _ in range(100)]
        a_count = results.count("a")

        # 'a' should appear less than average (20%) due to recency penalty
        # With 5 items, equal distribution would be ~20 each
        assert a_count < 40  # Should be well below 40%

    def test_discovery_bias_favors_new(self, engine):
        """Never-picked items should have discovery bonus."""
        pool = ["old_1", "old_2", "new_1", "new_2"]
        # Use old items heavily
        for _ in range(10):
            engine._log_selection("discovery_test", "old_1")
            engine._log_selection("discovery_test", "old_2")

        # Pick 200 times for statistical significance
        results = [engine.pick("discovery_test", pool) for _ in range(200)]
        new_count = results.count("new_1") + results.count("new_2")
        old_count = results.count("old_1") + results.count("old_2")
        # New items (weight 2.0 each) should appear more than old (weight 0.1 each)
        assert new_count > old_count
