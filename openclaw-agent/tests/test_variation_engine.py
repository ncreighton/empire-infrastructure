"""Tests for openclaw/forge/variation_engine.py — anti-repetition pool selection."""

import pytest

from openclaw.forge.variation_engine import VariationEngine


@pytest.fixture
def engine():
    return VariationEngine()


class TestVariationEngine:
    def test_pick_returns_item_from_pool(self, engine):
        pool = ["alpha", "beta", "gamma"]
        item = engine.pick(pool, category="test")
        assert item in pool

    def test_pick_empty_pool_returns_none(self, engine):
        assert engine.pick([], category="test") is None

    def test_pick_no_repeat_until_exhausted(self, engine):
        pool = ["a", "b", "c"]
        seen = set()
        for _ in range(3):
            item = engine.pick(pool, category="exhaust")
            assert item not in seen, f"Got repeat '{item}' before pool exhausted"
            seen.add(item)
        # All items should have been seen
        assert seen == {"a", "b", "c"}

    def test_pick_resets_after_exhaustion(self, engine):
        pool = ["x", "y"]
        # Exhaust the pool
        engine.pick(pool, category="cycle")
        engine.pick(pool, category="cycle")
        # Third pick should still work (resets internally)
        item = engine.pick(pool, category="cycle")
        assert item in pool

    def test_pick_n_returns_correct_count(self, engine):
        pool = list(range(10))
        result = engine.pick_n(pool, 5, category="n_test")
        assert len(result) == 5
        # All items should be unique
        assert len(set(result)) == 5

    def test_pick_n_exceeding_pool(self, engine):
        pool = ["a", "b"]
        result = engine.pick_n(pool, 10, category="over")
        assert len(result) == 2  # Capped at pool size

    def test_pick_n_empty_pool(self, engine):
        assert engine.pick_n([], 5, category="empty") == []

    def test_reset_specific_category(self, engine):
        pool = ["a", "b"]
        engine.pick(pool, category="cat1")
        engine.pick(pool, category="cat2")
        engine.reset(category="cat1")
        # cat1 is reset, cat2 still tracked
        # Pick from cat1 should have full pool available
        item = engine.pick(pool, category="cat1")
        assert item in pool

    def test_reset_all_categories(self, engine):
        pool = ["a", "b"]
        engine.pick(pool, category="x")
        engine.pick(pool, category="y")
        engine.reset()
        # Both categories should be cleared
        item_x = engine.pick(pool, category="x")
        item_y = engine.pick(pool, category="y")
        assert item_x in pool
        assert item_y in pool

    def test_categories_are_independent(self, engine):
        pool = ["a", "b"]
        # Pick 'a' from category "first"
        for _ in range(2):
            engine.pick(pool, category="first")
        # "second" should still have full pool
        seen = set()
        for _ in range(2):
            seen.add(engine.pick(pool, category="second"))
        assert len(seen) == 2

    def test_shuffle_pool_returns_copy(self, engine):
        pool = [1, 2, 3, 4, 5]
        shuffled = engine.shuffle_pool(pool)
        assert set(shuffled) == set(pool)
        assert len(shuffled) == len(pool)
        # Original is unmodified
        assert pool == [1, 2, 3, 4, 5]
