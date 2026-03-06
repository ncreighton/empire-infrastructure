"""Tests for openclaw/forge/market_oracle.py — MarketOracle prioritization."""

import pytest

from openclaw.forge.market_oracle import MarketOracle
from openclaw.models import OracleRecommendation, OraclePriority, PlatformCategory
from openclaw.knowledge.platforms import get_all_platform_ids


@pytest.fixture
def oracle():
    return MarketOracle()


class TestMarketOracle:
    def test_prioritize_returns_sorted_list(self, oracle):
        recs = oracle.prioritize_platforms()
        assert isinstance(recs, list)
        assert len(recs) >= 10
        # Verify descending order by score
        for i in range(len(recs) - 1):
            assert recs[i].score >= recs[i + 1].score

    def test_recommendations_are_correct_type(self, oracle):
        recs = oracle.prioritize_platforms()
        for rec in recs:
            assert isinstance(rec, OracleRecommendation)
            assert isinstance(rec.priority, OraclePriority)
            assert isinstance(rec.category, PlatformCategory)

    def test_completed_platforms_excluded(self, oracle):
        completed = {"gumroad", "etsy"}
        recs = oracle.prioritize_platforms(completed=completed)
        rec_ids = {r.platform_id for r in recs}
        assert "gumroad" not in rec_ids
        assert "etsy" not in rec_ids

    def test_all_completed_returns_empty(self, oracle):
        all_ids = set(get_all_platform_ids())
        recs = oracle.prioritize_platforms(completed=all_ids)
        assert recs == []

    def test_recommended_order_is_sequential(self, oracle):
        recs = oracle.prioritize_platforms()
        for i, rec in enumerate(recs):
            assert rec.recommended_order == i + 1

    def test_recommend_next_returns_top_pick(self, oracle):
        rec = oracle.recommend_next()
        assert rec is not None
        assert isinstance(rec, OracleRecommendation)
        assert rec.recommended_order == 1

    def test_recommend_next_excludes_completed(self, oracle):
        # Get top recommendation first
        top = oracle.recommend_next()
        assert top is not None
        # Now exclude it
        second = oracle.recommend_next(completed={top.platform_id})
        assert second is not None
        assert second.platform_id != top.platform_id

    def test_recommend_next_returns_none_when_all_done(self, oracle):
        all_ids = set(get_all_platform_ids())
        rec = oracle.recommend_next(completed=all_ids)
        assert rec is None

    def test_score_components_populated(self, oracle):
        recs = oracle.prioritize_platforms()
        for rec in recs[:5]:
            assert rec.monetization_score >= 0
            assert rec.audience_score >= 0
            assert rec.seo_score >= 0
            assert rec.effort_score >= 0
            assert rec.score > 0

    def test_reasoning_is_human_readable(self, oracle):
        recs = oracle.prioritize_platforms()
        for rec in recs[:5]:
            assert isinstance(rec.reasoning, str)
            assert len(rec.reasoning) > 20
            assert "score" in rec.reasoning.lower() or "Composite" in rec.reasoning

    def test_category_filter(self, oracle):
        recs = oracle.prioritize_platforms(
            categories={PlatformCategory.DIGITAL_PRODUCT}
        )
        assert len(recs) >= 1
        assert all(r.category == PlatformCategory.DIGITAL_PRODUCT for r in recs)

    def test_priority_assignment(self, oracle):
        recs = oracle.prioritize_platforms()
        # High-scoring platforms should have CRITICAL or HIGH priority
        if recs:
            top = recs[0]
            if top.score >= 80:
                assert top.priority == OraclePriority.CRITICAL
            elif top.score >= 60:
                assert top.priority == OraclePriority.HIGH

    def test_get_category_summary(self, oracle):
        summary = oracle.get_category_summary()
        assert isinstance(summary, dict)
        assert len(summary) >= 1
        for cat, data in summary.items():
            assert "total" in data
            assert "completed" in data
            assert "remaining" in data
            assert data["total"] >= 1
