"""Tests for the VariationEngine — anti-repetition and pool selection."""

import pytest
from grimoire.forge.variation_engine import (
    VariationEngine,
    AFFIRMATION_POOLS,
    ELEMENT_IMAGERY_POOLS,
    AFTERCARE_POOLS,
    PREPARATION_POOLS,
    CHALLENGE_POOLS,
    CHECKLIST_POOLS,
    AFTERCARE_AMPLIFY_POOLS,
    ETHICAL_NOTE_POOLS,
    DAILY_SUGGESTION_POOLS,
    QUICK_PRACTICE_POOLS,
    TIMING_ADVICE_POOLS,
)


@pytest.fixture
def engine():
    """Create a VariationEngine with in-memory DB for tests."""
    return VariationEngine(db_path=":memory:")


# ── Basic functionality ──────────────────────────────────────────────────


class TestPick:
    def test_pick_returns_item_from_pool(self, engine):
        pool = ["alpha", "beta", "gamma"]
        result = engine.pick("test_pool", pool)
        assert result in pool

    def test_pick_empty_pool_returns_empty(self, engine):
        assert engine.pick("empty", []) == ""

    def test_pick_single_item_returns_it(self, engine):
        assert engine.pick("single", ["only"]) == "only"

    def test_pick_logs_selection(self, engine):
        engine.pick("logged", ["a", "b", "c"])
        with engine._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM recency_log WHERE pool_name = 'logged'"
            ).fetchone()["c"]
        assert count == 1


class TestPickN:
    def test_pick_n_returns_correct_count(self, engine):
        pool = ["a", "b", "c", "d", "e"]
        result = engine.pick_n("multi", pool, 3)
        assert len(result) == 3

    def test_pick_n_returns_unique_items(self, engine):
        pool = ["a", "b", "c", "d", "e"]
        result = engine.pick_n("unique", pool, 4)
        assert len(set(result)) == 4

    def test_pick_n_all_from_pool(self, engine):
        pool = ["a", "b", "c", "d", "e"]
        result = engine.pick_n("all", pool, 5)
        assert set(result) == set(pool)

    def test_pick_n_caps_at_pool_size(self, engine):
        pool = ["x", "y"]
        result = engine.pick_n("small", pool, 10)
        assert len(result) == 2

    def test_pick_n_empty_pool(self, engine):
        assert engine.pick_n("empty", [], 3) == []


# ── Anti-repetition ──────────────────────────────────────────────────────


class TestAntiRepetition:
    def test_never_used_items_get_discovery_weight(self, engine):
        pool = ["a", "b", "c"]
        weights = engine._get_recency_weights("fresh", pool)
        assert all(w == 2.0 for w in weights)

    def test_recently_used_items_get_low_weight(self, engine):
        pool = ["a", "b", "c"]
        # Use "a" once
        engine._log_selection("recent", "a")
        weights = engine._get_recency_weights("recent", pool)
        # "a" should have weight 0.1, others 2.0
        assert weights[0] == 0.1
        assert weights[1] == 2.0
        assert weights[2] == 2.0

    def test_variation_across_repeated_calls(self, engine):
        """Calling pick 10 times on a 5-item pool should select at least 3 different items."""
        pool = ["a", "b", "c", "d", "e"]
        seen = set()
        for _ in range(10):
            seen.add(engine.pick("variation", pool))
        assert len(seen) >= 3

    def test_pool_exhaustion_still_returns(self, engine):
        """When all items were recently used, pick still returns something."""
        pool = ["x", "y"]
        # Use both items
        engine._log_selection("exhausted", "x")
        engine._log_selection("exhausted", "y")
        result = engine.pick("exhausted", pool)
        assert result in pool


# ── Pool data integrity ──────────────────────────────────────────────────


class TestPoolIntegrity:
    def test_affirmation_pools_have_multiple_per_intention(self):
        for intention, variants in AFFIRMATION_POOLS.items():
            assert len(variants) >= 4, f"{intention} has too few affirmations"

    def test_element_imagery_pools_have_multiple(self):
        for element, variants in ELEMENT_IMAGERY_POOLS.items():
            assert len(variants) >= 3, f"{element} has too few imagery"

    def test_aftercare_pools_not_empty(self):
        assert len(AFTERCARE_POOLS) >= 5

    def test_preparation_pools_not_empty(self):
        assert len(PREPARATION_POOLS) >= 5

    def test_challenge_pools_sufficient(self):
        assert len(CHALLENGE_POOLS) >= 12

    def test_checklist_pools_sufficient(self):
        assert len(CHECKLIST_POOLS) >= 12

    def test_aftercare_amplify_pools_sufficient(self):
        assert len(AFTERCARE_AMPLIFY_POOLS) >= 10

    def test_ethical_note_pools_sufficient(self):
        assert len(ETHICAL_NOTE_POOLS) >= 8

    def test_daily_suggestion_pools_cover_planets(self):
        expected_planets = {"sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn"}
        actual_keys = {k.lower() for k in DAILY_SUGGESTION_POOLS.keys()}
        assert expected_planets.issubset(actual_keys)

    def test_quick_practice_pools_cover_planets(self):
        expected_planets = {"sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn"}
        actual_keys = {k.lower() for k in QUICK_PRACTICE_POOLS.keys()}
        assert expected_planets.issubset(actual_keys)

    def test_timing_advice_pools_have_multiple(self):
        for intention, variants in TIMING_ADVICE_POOLS.items():
            assert len(variants) >= 2, f"{intention} has too few timing advice"
