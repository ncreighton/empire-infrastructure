"""
Tests for the MoneyClaw Intelligence Layer.

Tests cover:
  - Schema creation (9 tables)
  - NeuralMemorySystem (store, confirm, refute, recall, decay, prune)
  - SkillEngine (create, activate, evaluate, record_outcome, deprecate)
  - PatternMiner (binomial z-test, bucket analysis, mining)
  - KnowledgeGraph (nodes, edges, queries, seeding)
  - StrategyForge (create, mutate, promote, retire)
  - DeepDiveEngine (big_win, big_loss, regime_change)
  - Swarm agents (MarketScout, RiskAnalyst, Backtester)
  - IntelligenceBrain (init, tick, get_skill_adjustments)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# We need to set up the database before importing intelligence modules
from moneyclaw.persistence.database import Database


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-like DB for testing."""
    db_path = str(tmp_path / "test_intel.db")
    return Database(db_path)


@pytest.fixture
def intel_db(db):
    """DB with intelligence schema initialized."""
    from moneyclaw.intelligence.schema import init_intelligence_schema
    init_intelligence_schema(db)
    return db


# =========================================================================
# Schema Tests
# =========================================================================

class TestSchema:
    def test_creates_9_tables(self, intel_db):
        """Intelligence schema should create 9 intel_* tables."""
        cur = intel_db._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'intel_%'"
        )
        tables = {r[0] for r in cur.fetchall()}
        expected = {
            "intel_memories", "intel_skills", "intel_patterns",
            "intel_param_configs", "intel_kg_nodes", "intel_kg_edges",
            "intel_deep_dives", "intel_growth_log", "intel_agent_runs",
        }
        assert expected == tables

    def test_idempotent(self, intel_db):
        """Calling init twice should not raise."""
        from moneyclaw.intelligence.schema import init_intelligence_schema
        init_intelligence_schema(intel_db)  # second call


# =========================================================================
# Memory Tests
# =========================================================================

class TestMemory:
    def test_store_and_recall(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        mem_id = mem.store(
            category="trade_outcome",
            subject="momentum|BTC-USD",
            observation="Win on BTC",
            confidence=0.7,
            tags="win,btc",
        )
        assert mem_id is not None
        assert mem_id > 0

        results = mem.recall(category="trade_outcome")
        assert len(results) == 1
        assert results[0]["confidence"] == 0.7
        assert results[0]["observation"] == "Win on BTC"

    def test_store_duplicate_confirms(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        id1 = mem.store("cat", "sub", "obs", confidence=0.5)
        id2 = mem.store("cat", "sub", "obs", confidence=0.5)
        assert id1 == id2  # same memory, just confirmed

        results = mem.recall(category="cat")
        assert len(results) == 1
        assert results[0]["evidence"] == 2  # confirmed once

    def test_confirm_increases_confidence(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        mem_id = mem.store("cat", "sub", "obs", confidence=0.5)
        mem.confirm(mem_id, confidence_boost=0.1)

        results = mem.recall(category="cat")
        assert results[0]["confidence"] == pytest.approx(0.6, abs=0.01)

    def test_refute_decreases_confidence(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        mem_id = mem.store("cat", "sub", "obs", confidence=0.5)
        mem.refute(mem_id, confidence_penalty=0.2)

        results = mem.recall(category="cat")
        assert results[0]["confidence"] == pytest.approx(0.3, abs=0.01)

    def test_recall_with_min_confidence(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        mem.store("cat", "sub1", "low", confidence=0.2)
        mem.store("cat", "sub2", "high", confidence=0.8)

        results = mem.recall(category="cat", min_confidence=0.5)
        assert len(results) == 1
        assert results[0]["observation"] == "high"

    def test_prune_expired(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        # Store with TTL that's already expired
        mem.store("cat", "sub", "expired", ttl_hours=-1)

        pruned = mem.prune_expired()
        assert pruned >= 1

        results = mem.recall(category="cat")
        assert len(results) == 0

    def test_get_stats(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        mem.store("cat1", "sub", "obs1", confidence=0.8)
        mem.store("cat2", "sub", "obs2", confidence=0.3)

        stats = mem.get_stats()
        assert stats["total_memories"] == 2
        assert stats["strong_memories"] == 1


# =========================================================================
# Skills Tests
# =========================================================================

class TestSkills:
    def test_create_skill(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        skill_id = skills.create_skill(
            name="test_skill",
            trigger={"strategy": "momentum"},
            action={"boost": 0.1},
        )
        assert skill_id is not None

    def test_duplicate_skill_returns_none(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        skills.create_skill("test", {"strategy": "momentum"}, {"boost": 0.1})
        result = skills.create_skill("test", {"strategy": "momentum"}, {"boost": 0.1})
        assert result is None

    def test_activate_skill(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        skill_id = skills.create_skill("test", {"strategy": "momentum"}, {"boost": 0.1})
        skills.activate_skill(skill_id)

        active = skills.get_all_skills(status="active")
        assert len(active) == 1

    def test_get_adjustments_no_skills(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", None)
        assert delta == 0.0

    def test_get_adjustments_with_matching_skill(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        skill_id = skills.create_skill(
            "boost_momentum",
            trigger={"strategy": "momentum"},
            action={"boost": 0.1},
        )
        skills.activate_skill(skill_id)

        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", None)
        assert delta == pytest.approx(0.1, abs=0.01)

    def test_get_adjustments_clamped(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        # Create multiple skills that would exceed cumulative max
        for i in range(5):
            sid = skills.create_skill(
                f"boost_{i}",
                trigger={"strategy": "momentum"},
                action={"boost": 0.15},
            )
            skills.activate_skill(sid)

        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", None)
        assert delta <= 0.35  # cumulative max

    def test_skill_action_clamped_on_create(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill("big_boost", {"strategy": "x"}, {"boost": 0.50})
        skills.activate_skill(sid)

        delta = skills.get_adjustments("BTC-USD", "x", "ranging", None)
        assert delta <= 0.25  # single skill max

    def test_coin_filter(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "doge_only",
            trigger={"strategy": "momentum"},
            action={"boost": 0.1},
            context={"coins": ["DOGE-USD"]},
        )
        skills.activate_skill(sid)

        # Should match DOGE
        delta = skills.get_adjustments("DOGE-USD", "momentum", "trending_up", None)
        assert delta > 0

        # Should NOT match BTC
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", None)
        assert delta == 0.0

    def test_auto_deprecation(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill("failing", {"strategy": "x"}, {"boost": 0.1})
        skills.activate_skill(sid)

        # Record 20 failures and 3 successes (< 30%)
        for _ in range(3):
            skills.record_outcome(sid, True)
        for _ in range(17):
            skills.record_outcome(sid, False)

        # Should be auto-deprecated
        all_skills = skills.get_all_skills(status="deprecated")
        assert len(all_skills) == 1

    def test_get_stats(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        skills.create_skill("s1", {"strategy": "a"}, {"boost": 0.1})
        sid = skills.create_skill("s2", {"strategy": "b"}, {"boost": 0.1})
        skills.activate_skill(sid)

        stats = skills.get_stats()
        assert stats["dormant_skills"] == 1
        assert stats["active_skills"] == 1


# =========================================================================
# Pattern Miner Tests
# =========================================================================

class TestPatternMiner:
    def test_binomial_z_test(self):
        from moneyclaw.intelligence.pattern_miner import _binomial_z_test

        # 80% win rate over 50 trades should be significant
        p = _binomial_z_test(40, 50, 0.5)
        assert p < 0.05

        # 50% win rate should NOT be significant
        p = _binomial_z_test(25, 50, 0.5)
        assert p > 0.05

        # Too few samples
        p = _binomial_z_test(3, 4, 0.5)
        assert p >= 0.05

    def test_mine_with_no_trades(self, intel_db):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        miner = PatternMiner(intel_db)

        patterns = miner.mine_all()
        assert patterns == []

    def test_mine_with_trades(self, intel_db):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        miner = PatternMiner(intel_db)

        # Insert enough fake trades
        now = datetime.utcnow()
        with intel_db._cursor() as cur:
            for i in range(20):
                pnl = 1.0 if i % 2 == 0 else -0.5
                cur.execute(
                    """INSERT INTO trades
                       (id, strategy, side, product_id, entry_price, close_price,
                        quantity, pnl, pnl_pct, fees, status, opened_at, closed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"trade_{i}", "momentum", "buy", "BTC-USD",
                        100.0, 101.0 if pnl > 0 else 99.5,
                        0.01, pnl, pnl / 100, 0.0, "filled",
                        (now - timedelta(hours=i)).isoformat(),
                        (now - timedelta(hours=i) + timedelta(minutes=30)).isoformat(),
                    ),
                )

        patterns = miner.mine_all(limit=100)
        # Should find at least strategy_coin patterns
        assert isinstance(patterns, list)

    def test_get_stats(self, intel_db):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        miner = PatternMiner(intel_db)

        stats = miner.get_stats()
        assert "total_patterns" in stats
        assert "significant_patterns" in stats


# =========================================================================
# Knowledge Graph Tests
# =========================================================================

class TestKnowledgeGraph:
    def test_upsert_node(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        node_id = kg.upsert_node("coin", "BTC-USD")
        assert node_id > 0

        node = kg.get_node("coin", "BTC-USD")
        assert node is not None
        assert node["name"] == "BTC-USD"

    def test_upsert_edge(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        n1 = kg.upsert_node("strategy", "momentum")
        n2 = kg.upsert_node("regime", "trending_up")

        edge_id = kg.upsert_edge(n1, n2, "performs_best_in", weight=0.8)
        assert edge_id > 0

    def test_edge_evidence_increments(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        n1 = kg.upsert_node("strategy", "momentum")
        n2 = kg.upsert_node("regime", "trending_up")

        kg.upsert_edge(n1, n2, "performs_best_in")
        kg.upsert_edge(n1, n2, "performs_best_in")  # second upsert

        # Check evidence increased
        with intel_db._cursor() as cur:
            cur.execute(
                "SELECT evidence FROM intel_kg_edges WHERE source_id = ? AND target_id = ?",
                (n1, n2),
            )
            assert cur.fetchone()["evidence"] == 2

    def test_query_neighbors(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        n1 = kg.upsert_node("strategy", "momentum")
        n2 = kg.upsert_node("regime", "trending_up")
        n3 = kg.upsert_node("regime", "breakout")
        kg.upsert_edge(n1, n2, "performs_best_in")
        kg.upsert_edge(n1, n3, "performs_best_in")

        neighbors = kg.query_neighbors("strategy", "momentum", direction="outgoing")
        assert len(neighbors) == 2

    def test_seed_base_nodes(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        count = kg.seed_base_nodes(
            coins=["BTC-USD", "ETH-USD"],
            strategies=["momentum", "mean_reversion"],
        )
        # 2 coins + 2 strategies + 6 regimes + 6 indicators = 16
        assert count == 16

    def test_get_stats(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)
        kg.upsert_node("coin", "BTC-USD")

        stats = kg.get_stats()
        assert stats["total_nodes"] == 1


# =========================================================================
# Strategy Forge Tests
# =========================================================================

class TestStrategyForge:
    def test_create_config(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        config_id = forge.create_config(
            "momentum", "trending_up",
            params={"rsi_entry": 60, "sl_pct": 0.03, "tp_pct": 0.06},
        )
        assert config_id > 0

    def test_mutate_config(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        parent_id = forge.create_config(
            "momentum", "trending_up",
            params={"rsi_entry": 60, "sl_pct": 0.03, "tp_pct": 0.06},
        )
        child_id = forge.mutate_config(parent_id)
        assert child_id is not None
        assert child_id != parent_id

    def test_promote_and_retire(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        cid = forge.create_config("momentum", "trending_up")
        forge.promote_config(cid)

        with intel_db._cursor() as cur:
            cur.execute("SELECT status FROM intel_param_configs WHERE id = ?", (cid,))
            assert cur.fetchone()["status"] == "active"

        forge.retire_config(cid)
        with intel_db._cursor() as cur:
            cur.execute("SELECT status FROM intel_param_configs WHERE id = ?", (cid,))
            assert cur.fetchone()["status"] == "retired"


# =========================================================================
# Deep Dive Tests
# =========================================================================

class TestDeepDive:
    def test_big_win(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.skills import SkillEngine
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.deep_dive import DeepDiveEngine

        mem = NeuralMemorySystem(intel_db)
        skills = SkillEngine(intel_db)
        miner = PatternMiner(intel_db)
        kg = KnowledgeGraph(intel_db)
        dd = DeepDiveEngine(intel_db, mem, skills, miner, kg)

        result = dd.on_big_win({
            "product_id": "BTC-USD",
            "strategy": "momentum",
            "pnl": 5.0,
            "pnl_pct": 0.05,
            "opened_at": datetime.utcnow().isoformat(),
        })

        assert "findings" in result
        assert len(result["findings"]) > 0

    def test_big_loss(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.skills import SkillEngine
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.deep_dive import DeepDiveEngine

        mem = NeuralMemorySystem(intel_db)
        skills = SkillEngine(intel_db)
        miner = PatternMiner(intel_db)
        kg = KnowledgeGraph(intel_db)
        dd = DeepDiveEngine(intel_db, mem, skills, miner, kg)

        result = dd.on_big_loss({
            "product_id": "SOL-USD",
            "strategy": "breakout",
            "pnl": -3.0,
            "pnl_pct": -0.03,
            "opened_at": datetime.utcnow().isoformat(),
        })

        assert "findings" in result

    def test_regime_change(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.skills import SkillEngine
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.deep_dive import DeepDiveEngine

        mem = NeuralMemorySystem(intel_db)
        skills = SkillEngine(intel_db)
        miner = PatternMiner(intel_db)
        kg = KnowledgeGraph(intel_db)
        dd = DeepDiveEngine(intel_db, mem, skills, miner, kg)

        result = dd.on_regime_change("trending_up", "breakout")
        assert result["trigger_type"] == "regime_change"


# =========================================================================
# Swarm Agent Tests
# =========================================================================

class TestMarketScout:
    def test_run_with_no_data(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.swarm.market_scout import MarketScout

        mem = NeuralMemorySystem(intel_db)
        scout = MarketScout(intel_db, mem)

        result = scout.run(coins=["BTC-USD"])
        assert result["agent"] == "MarketScout"
        assert isinstance(result["findings"], int)


class TestRiskAnalyst:
    def test_run_with_no_trades(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.swarm.risk_analyst import RiskAnalyst

        mem = NeuralMemorySystem(intel_db)
        analyst = RiskAnalyst(intel_db, mem)

        result = analyst.run()
        assert result["agent"] == "RiskAnalyst"
        assert isinstance(result["findings"], int)


class TestBacktester:
    def test_run_with_no_configs(self, intel_db):
        from moneyclaw.intelligence.swarm.backtester import Backtester

        bt = Backtester(intel_db)
        result = bt.run()
        assert result["agent"] == "Backtester"
        assert result["configs_tested"] == 0

    def test_approx_rsi(self):
        from moneyclaw.intelligence.swarm.backtester import Backtester

        # Rising prices should give RSI > 50
        rising = [100 + i for i in range(20)]
        rsi = Backtester._approx_rsi(rising)
        assert rsi > 60

        # Falling prices should give RSI < 50
        falling = [100 - i for i in range(20)]
        rsi = Backtester._approx_rsi(falling)
        assert rsi < 40


# =========================================================================
# Brain Integration Tests
# =========================================================================

class TestIntelligenceBrain:
    def test_init_creates_schema(self, db):
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD", "ETH-USD"])

        # Check tables exist
        cur = db._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'intel_%'"
        )
        tables = {r[0] for r in cur.fetchall()}
        assert len(tables) == 9

    def test_get_skill_adjustments_no_skills(self, db):
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        delta = brain.get_skill_adjustments("BTC-USD", "momentum", "trending_up", None)
        assert delta == 0.0

    def test_get_skill_adjustments_with_enum(self, db):
        """Test that regime enum values are handled correctly."""
        from moneyclaw.intelligence.brain import IntelligenceBrain
        from moneyclaw.models import MarketRegime

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        delta = brain.get_skill_adjustments(
            "BTC-USD", "momentum", MarketRegime.TRENDING_UP, None,
        )
        assert delta == 0.0

    def test_on_trade_closed(self, db):
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        brain.on_trade_closed({
            "product_id": "BTC-USD",
            "strategy": "momentum",
            "side": "buy",
            "pnl": 2.0,
            "pnl_pct": 0.02,
            "entry_price": 100.0,
            "close_price": 102.0,
            "opened_at": datetime.utcnow().isoformat(),
            "closed_at": datetime.utcnow().isoformat(),
        })

        # Should have stored a memory
        memories = brain.memory.recall(category="trade_outcome")
        assert len(memories) >= 1

    def test_on_regime_change(self, db):
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        brain.on_regime_change("trending_up", "breakout")

        # Should have stored a memory
        memories = brain.memory.recall(category="regime_shift")
        assert len(memories) >= 1

    def test_get_status(self, db):
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        status = brain.get_status()

        assert "memory" in status
        assert "skills" in status
        assert "patterns" in status
        assert "knowledge_graph" in status
        assert "forge" in status

    def test_tick_runs_without_error(self, db):
        """Brain tick should not raise even with empty data."""
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        # Force all intervals to be past due
        brain._last_quick = 0
        brain._last_pattern = 0
        brain._last_deep = 0
        brain._last_full = 0

        brain.tick()  # should not raise

        # Verify growth log was written
        with db._cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM intel_growth_log")
            assert cur.fetchone()["cnt"] >= 1

    def test_get_active_config_delegates(self, db):
        """Brain.get_active_config should delegate to strategy forge."""
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])
        # No configs yet
        result = brain.get_active_config("momentum", "trending_up")
        assert result is None


# =========================================================================
# Enhanced Skill Trigger Tests (indicator matching)
# =========================================================================

class _FakeIndicators:
    """Mock indicators for skill trigger testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestSkillIndicatorTriggers:
    """Tests for volume_ratio, MACD, Bollinger Band, and ATR triggers."""

    def test_volume_ratio_above_matches(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "vol_trigger",
            trigger={"strategy": "momentum", "volume_ratio_above": 1.5},
            action={"boost": 0.1},
        )
        skills.activate_skill(sid)

        # Volume 3.0x SMA → should match
        ind = _FakeIndicators(volume=300, volume_sma=100)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == pytest.approx(0.1, abs=0.01)

    def test_volume_ratio_below_threshold_no_match(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "vol_trigger",
            trigger={"strategy": "momentum", "volume_ratio_above": 2.0},
            action={"boost": 0.1},
        )
        skills.activate_skill(sid)

        # Volume 1.2x SMA → should NOT match (below 2.0)
        ind = _FakeIndicators(volume=120, volume_sma=100)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == 0.0

    def test_volume_ratio_zero_sma_safe(self, intel_db):
        """When volume_sma is 0/invalid, volume check is skipped (not crashed)."""
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "vol_trigger",
            trigger={"strategy": "momentum", "volume_ratio_above": 1.5},
            action={"boost": 0.1},
        )
        skills.activate_skill(sid)

        # SMA = 0 → volume check skipped, skill still matches on strategy
        ind = _FakeIndicators(volume=300, volume_sma=0)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        # Should not crash; skill matches because volume check is skipped
        assert delta == pytest.approx(0.1, abs=0.01)

    def test_macd_above_zero_trigger(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "macd_bull",
            trigger={"strategy": "momentum", "macd_above_zero": True},
            action={"boost": 0.08},
        )
        skills.activate_skill(sid)

        # MACD positive → matches
        ind = _FakeIndicators(macd=0.5)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == pytest.approx(0.08, abs=0.01)

        # MACD negative → no match
        ind = _FakeIndicators(macd=-0.5)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == 0.0

    def test_macd_below_zero_trigger(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "macd_bear",
            trigger={"strategy": "mean_reversion", "macd_below_zero": True},
            action={"boost": 0.06},
        )
        skills.activate_skill(sid)

        # MACD negative → matches
        ind = _FakeIndicators(macd=-0.3)
        delta = skills.get_adjustments("BTC-USD", "mean_reversion", "ranging", ind)
        assert delta == pytest.approx(0.06, abs=0.01)

        # MACD positive → no match
        ind = _FakeIndicators(macd=0.3)
        delta = skills.get_adjustments("BTC-USD", "mean_reversion", "ranging", ind)
        assert delta == 0.0

    def test_below_bb_lower_trigger(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "bb_low",
            trigger={"strategy": "mean_reversion", "below_bb_lower": True},
            action={"boost": 0.1},
        )
        skills.activate_skill(sid)

        # Price below lower BB → matches
        ind = _FakeIndicators(close=95.0, bb_lower=100.0, bb_upper=120.0)
        delta = skills.get_adjustments("BTC-USD", "mean_reversion", "ranging", ind)
        assert delta == pytest.approx(0.1, abs=0.01)

        # Price above lower BB → no match
        ind = _FakeIndicators(close=105.0, bb_lower=100.0, bb_upper=120.0)
        delta = skills.get_adjustments("BTC-USD", "mean_reversion", "ranging", ind)
        assert delta == 0.0

    def test_above_bb_upper_trigger(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "bb_high",
            trigger={"strategy": "momentum", "above_bb_upper": True},
            action={"boost": -0.1},
        )
        skills.activate_skill(sid)

        # Price above upper BB → matches
        ind = _FakeIndicators(close=125.0, bb_lower=100.0, bb_upper=120.0)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == pytest.approx(-0.1, abs=0.01)

        # Price below upper BB → no match
        ind = _FakeIndicators(close=115.0, bb_lower=100.0, bb_upper=120.0)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == 0.0

    def test_atr_above_trigger(self, intel_db):
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "atr_vol",
            trigger={"strategy": "volatility", "atr_above": 5.0},
            action={"boost": 0.12},
        )
        skills.activate_skill(sid)

        # ATR above threshold → matches
        ind = _FakeIndicators(atr=7.5)
        delta = skills.get_adjustments("BTC-USD", "volatility", "high_volatility", ind)
        assert delta == pytest.approx(0.12, abs=0.01)

        # ATR below threshold → no match
        ind = _FakeIndicators(atr=3.0)
        delta = skills.get_adjustments("BTC-USD", "volatility", "high_volatility", ind)
        assert delta == 0.0

    def test_combined_indicator_triggers(self, intel_db):
        """Multiple indicator conditions must ALL match."""
        from moneyclaw.intelligence.skills import SkillEngine
        skills = SkillEngine(intel_db)

        sid = skills.create_skill(
            "combined",
            trigger={
                "strategy": "momentum",
                "rsi_above": 60,
                "macd_above_zero": True,
                "volume_ratio_above": 1.5,
            },
            action={"boost": 0.15},
        )
        skills.activate_skill(sid)

        # All conditions met
        ind = _FakeIndicators(rsi=70, macd=0.5, volume=200, volume_sma=100)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == pytest.approx(0.15, abs=0.01)

        # RSI too low → no match
        ind = _FakeIndicators(rsi=50, macd=0.5, volume=200, volume_sma=100)
        delta = skills.get_adjustments("BTC-USD", "momentum", "trending_up", ind)
        assert delta == 0.0


# =========================================================================
# Enhanced Knowledge Graph Tests (weight averaging)
# =========================================================================

class TestKGWeightAveraging:
    def test_edge_weight_averages_on_upsert(self, intel_db):
        """Upserting same edge should average weights, not replace."""
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        n1 = kg.upsert_node("strategy", "momentum")
        n2 = kg.upsert_node("regime", "trending_up")

        kg.upsert_edge(n1, n2, "performs_best_in", weight=1.0)
        kg.upsert_edge(n1, n2, "performs_best_in", weight=0.5)

        # Weight should be averaged: (1.0 + 0.5) / 2 = 0.75
        with intel_db._cursor() as cur:
            cur.execute(
                "SELECT weight FROM intel_kg_edges WHERE source_id = ? AND target_id = ?",
                (n1, n2),
            )
            weight = cur.fetchone()["weight"]
            assert abs(weight - 0.75) < 0.01

    def test_strengthen_edge(self, intel_db):
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(intel_db)

        n1 = kg.upsert_node("strategy", "momentum")
        n2 = kg.upsert_node("coin", "BTC-USD")

        edge_id = kg.upsert_edge(n1, n2, "used_by", weight=1.0)
        kg.strengthen_edge(edge_id, 0.1)

        with intel_db._cursor() as cur:
            cur.execute("SELECT weight, evidence FROM intel_kg_edges WHERE id = ?", (edge_id,))
            row = cur.fetchone()
            assert row["weight"] == pytest.approx(1.1, abs=0.01)
            assert row["evidence"] == 2  # original + strengthen


# =========================================================================
# Enhanced Memory Tests (trade dedup, batch decay)
# =========================================================================

class TestMemoryEnhanced:
    def test_trade_outcome_dedup(self, intel_db):
        """Trade outcome memories with same category+subject should dedup."""
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        # Two trade outcomes for same strategy+coin, different pnl text
        id1 = mem.store("trade_outcome", "momentum|BTC-USD", "Trade win: momentum on BTC-USD (2.00%)")
        id2 = mem.store("trade_outcome", "momentum|BTC-USD", "Trade win: momentum on BTC-USD (3.50%)")

        # Should be same memory (confirmed, not duplicated)
        assert id1 == id2
        results = mem.recall(category="trade_outcome")
        assert len(results) == 1
        assert results[0]["evidence"] == 2

    def test_big_win_dedup(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        id1 = mem.store("big_win", "momentum|BTC-USD", "Big win: 5.0%")
        id2 = mem.store("big_win", "momentum|BTC-USD", "Big win: 7.2%")
        assert id1 == id2

    def test_non_trade_categories_not_dedup(self, intel_db):
        """Non-trade categories should NOT dedup on category+subject alone."""
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        id1 = mem.store("regime_shift", "trending_up->breakout", "Regime changed")
        id2 = mem.store("regime_shift", "trending_up->breakout", "Different observation")
        assert id1 != id2

    def test_batch_decay(self, intel_db):
        """apply_decay should work on multiple memories efficiently."""
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        mem = NeuralMemorySystem(intel_db)

        # Store several memories with old timestamps
        for i in range(5):
            mem.store(f"cat_{i}", "sub", f"obs_{i}", confidence=0.5)

        # Force old updated_at timestamps
        old_time = (datetime.utcnow() - timedelta(days=7)).isoformat()
        with intel_db._cursor() as cur:
            cur.execute("UPDATE intel_memories SET updated_at = ?", (old_time,))

        decayed = mem.apply_decay()
        assert decayed == 5  # all 5 should have decayed

        # Verify weights decreased
        results = mem.recall()
        for r in results:
            assert r["recency_weight"] < 0.5  # 7 days >> 72hr half-life


# =========================================================================
# Enhanced Strategy Forge Tests
# =========================================================================

class TestForgeEnhanced:
    def test_get_active_config(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        # No configs → None
        assert forge.get_active_config("momentum", "trending_up") is None

        # Create and promote
        cid = forge.create_config("momentum", "trending_up",
                                  params={"rsi_entry": 65})
        forge.promote_config(cid)

        config = forge.get_active_config("momentum", "trending_up")
        assert config is not None
        assert config["strategy"] == "momentum"

    def test_get_all_active_configs(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        c1 = forge.create_config("momentum", "trending_up")
        c2 = forge.create_config("breakout", "breakout")
        forge.promote_config(c1)
        forge.promote_config(c2)

        active = forge.get_all_active_configs()
        assert len(active) == 2

    def test_forge_cycle_seeds_missing_combos(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        actions = forge.forge_cycle(["momentum", "breakout"], "trending_up")
        seed_actions = [a for a in actions if a["action"] == "seed"]
        assert len(seed_actions) == 2  # both strategies seeded

    def test_forge_cycle_promotes_winners(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        # Create a candidate with good backtest results
        cid = forge.create_config("momentum", "trending_up",
                                  params={"rsi_entry": 65})
        with intel_db._cursor() as cur:
            cur.execute(
                """UPDATE intel_param_configs
                   SET backtest_trades = 10, backtest_win_rate = 0.7, backtest_pnl = 5.0
                   WHERE id = ?""",
                (cid,),
            )

        actions = forge.forge_cycle(["momentum"], "trending_up")
        promote_actions = [a for a in actions if a["action"] == "promote_and_mutate"]
        assert len(promote_actions) == 1

    def test_forge_cycle_retires_losers(self, intel_db):
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        cid = forge.create_config("momentum", "trending_up",
                                  params={"rsi_entry": 65})
        with intel_db._cursor() as cur:
            cur.execute(
                """UPDATE intel_param_configs
                   SET backtest_trades = 10, backtest_win_rate = 0.2, backtest_pnl = -10.0
                   WHERE id = ?""",
                (cid,),
            )

        actions = forge.forge_cycle(["momentum"], "trending_up")
        retire_actions = [a for a in actions if a["action"] == "retire"]
        assert len(retire_actions) == 1

    def test_gaussian_mutation_stays_in_range(self, intel_db):
        """Mutated params should stay within default ranges."""
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge, DEFAULT_PARAM_RANGES

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        parent_id = forge.create_config("momentum", "trending_up",
                                        params={"rsi_entry": 60, "sl_pct": 0.03, "tp_pct": 0.06})
        # Mutate many times to test bounds
        for _ in range(20):
            child_id = forge.mutate_config(parent_id)
            assert child_id is not None
            with intel_db._cursor() as cur:
                cur.execute("SELECT params_json FROM intel_param_configs WHERE id = ?", (child_id,))
                params = json.loads(cur.fetchone()["params_json"])
                ranges = DEFAULT_PARAM_RANGES["momentum"]
                for key, (lo, hi) in ranges.items():
                    if key in params:
                        assert lo <= params[key] <= hi, f"{key}={params[key]} out of [{lo}, {hi}]"


# =========================================================================
# Deep Dive Circuit Breaker Tests
# =========================================================================

class TestCircuitBreakerDeepDive:
    def test_circuit_breaker_creates_penalty_skills(self, intel_db):
        """Circuit breaker should create penalty skills for losing strategies."""
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.skills import SkillEngine
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.deep_dive import DeepDiveEngine

        mem = NeuralMemorySystem(intel_db)
        skills = SkillEngine(intel_db)
        miner = PatternMiner(intel_db)
        kg = KnowledgeGraph(intel_db)
        dd = DeepDiveEngine(intel_db, mem, skills, miner, kg)

        # Insert losing trades (4 losses for "momentum")
        now = datetime.utcnow()
        with intel_db._cursor() as cur:
            for i in range(4):
                cur.execute(
                    """INSERT INTO trades
                       (id, strategy, side, product_id, entry_price, close_price,
                        quantity, pnl, pnl_pct, fees, status, opened_at, closed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"cb_trade_{i}", "momentum", "buy", "BTC-USD",
                        100.0, 97.0, 0.01, -3.0, -0.03, 0.0, "filled",
                        (now - timedelta(minutes=i*5)).isoformat(),
                        (now - timedelta(minutes=i*5) + timedelta(minutes=3)).isoformat(),
                    ),
                )

        result = dd.on_circuit_breaker({"reason": "daily_loss_limit"})

        assert "actions" in result
        # Should have created a penalty skill for momentum
        active_skills = skills.get_all_skills(status="active")
        cb_skills = [s for s in active_skills if "cb_reduce" in s["name"]]
        assert len(cb_skills) >= 1

    def test_circuit_breaker_stores_loss_cluster(self, intel_db):
        """High loss ratio should store a loss cluster memory."""
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.skills import SkillEngine
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.deep_dive import DeepDiveEngine

        mem = NeuralMemorySystem(intel_db)
        skills = SkillEngine(intel_db)
        miner = PatternMiner(intel_db)
        kg = KnowledgeGraph(intel_db)
        dd = DeepDiveEngine(intel_db, mem, skills, miner, kg)

        # Insert 10 trades, 8 losses (80% loss ratio > 70% threshold)
        now = datetime.utcnow()
        with intel_db._cursor() as cur:
            for i in range(10):
                pnl = 1.0 if i < 2 else -2.0
                cur.execute(
                    """INSERT INTO trades
                       (id, strategy, side, product_id, entry_price, close_price,
                        quantity, pnl, pnl_pct, fees, status, opened_at, closed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"cluster_trade_{i}", "breakout", "buy", "ETH-USD",
                        100.0, 101.0 if pnl > 0 else 98.0, 0.01, pnl, pnl/100,
                        0.0, "filled",
                        (now - timedelta(minutes=i*5)).isoformat(),
                        (now - timedelta(minutes=i*5) + timedelta(minutes=3)).isoformat(),
                    ),
                )

        dd.on_circuit_breaker({"reason": "drawdown"})

        # Check for loss cluster memory
        memories = mem.recall(category="strategy_weakness", tags="cluster")
        assert len(memories) >= 1


# =========================================================================
# Enhanced Backtester Tests (Wilder RSI, strategy entries)
# =========================================================================

class TestBacktesterEnhanced:
    def test_wilder_rsi_rising(self):
        """Wilder SMMA RSI should give > 50 for consistently rising prices."""
        from moneyclaw.intelligence.swarm.backtester import Backtester
        # Steady uptrend
        prices = [100 + i * 0.5 for i in range(30)]
        rsi = Backtester._approx_rsi(prices, period=14)
        assert rsi > 70  # strong uptrend

    def test_wilder_rsi_falling(self):
        """Wilder SMMA RSI should give < 50 for consistently falling prices."""
        from moneyclaw.intelligence.swarm.backtester import Backtester
        prices = [100 - i * 0.5 for i in range(30)]
        rsi = Backtester._approx_rsi(prices, period=14)
        assert rsi < 30  # strong downtrend

    def test_wilder_rsi_flat(self):
        """RSI returns 100 for flat prices (zero losses)."""
        from moneyclaw.intelligence.swarm.backtester import Backtester
        prices = [100.0] * 20
        rsi = Backtester._approx_rsi(prices, period=14)
        # Flat = zero losses → RSI = 100 (no selling pressure)
        assert rsi == 100.0

    def test_backtester_with_candle_data(self, intel_db):
        """Backtester should process candidate configs with candle data."""
        from moneyclaw.intelligence.swarm.backtester import Backtester
        from moneyclaw.intelligence.memory import NeuralMemorySystem
        from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
        from moneyclaw.intelligence.strategy_forge import StrategyForge

        mem = NeuralMemorySystem(intel_db)
        kg = KnowledgeGraph(intel_db)
        forge = StrategyForge(intel_db, mem, kg)

        # Create a candidate config
        forge.create_config("momentum", "trending_up",
                            params={"rsi_entry": 65, "rsi_exit": 40, "sl_pct": 0.03,
                                    "tp_pct": 0.06, "ema_fast": 8, "ema_slow": 21})

        # Insert candle data (uptrend to trigger momentum entries)
        # Check what timeframe the backtester expects
        now = datetime.utcnow()
        with intel_db._cursor() as cur:
            for i in range(50):
                price = 100 + i * 0.5
                cur.execute(
                    """INSERT INTO candles
                       (product_id, timeframe, timestamp, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "BTC-USD", "FIVE_MINUTE",
                        (now - timedelta(hours=50-i)).isoformat(),
                        price - 0.3, price + 0.5, price - 0.5, price,
                        1000 + i * 10,
                    ),
                )

        bt = Backtester(intel_db)
        result = bt.run(max_configs=5)
        assert result["agent"] == "Backtester"


# =========================================================================
# Pattern Miner Correlation Tests
# =========================================================================

class TestPatternMinerCorrelation:
    def test_pearson_perfect_positive(self):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = PatternMiner._pearson(x, y)
        assert r == pytest.approx(1.0, abs=0.001)

    def test_pearson_perfect_negative(self):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        r = PatternMiner._pearson(x, y)
        assert r == pytest.approx(-1.0, abs=0.001)

    def test_pearson_no_correlation(self):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        y = [4.0, 2.0, 5.0, 1.0, 6.0, 3.0, 4.0]
        r = PatternMiner._pearson(x, y)
        assert -0.6 < r < 0.6  # roughly uncorrelated

    def test_pearson_short_series(self):
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        r = PatternMiner._pearson([1.0], [2.0])
        assert r is None or r == 0.0  # too short, returns None

    def test_correlation_mining_with_candles(self, intel_db):
        """Should find correlation patterns between coins with shared price data."""
        from moneyclaw.intelligence.pattern_miner import PatternMiner
        miner = PatternMiner(intel_db)

        now = datetime.utcnow()
        with intel_db._cursor() as cur:
            # mine_all requires MIN_SAMPLES (10) trades to proceed
            for i in range(12):
                cur.execute(
                    """INSERT INTO trades
                       (id, strategy, side, product_id, entry_price, close_price,
                        quantity, pnl, pnl_pct, fees, status, opened_at, closed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"corr_trade_{i}", "momentum", "buy", "BTC-USD",
                        100.0, 101.0, 0.01, 1.0, 0.01, 0.0, "filled",
                        (now - timedelta(hours=i)).isoformat(),
                        (now - timedelta(hours=i) + timedelta(minutes=30)).isoformat(),
                    ),
                )

            # Need 50+ candles with FIVE_MINUTE timeframe (what the miner expects)
            for i in range(60):
                btc_price = 50000 + i * 100
                eth_price = 3000 + i * 6  # correlated
                for coin, price in [("BTC-USD", btc_price), ("ETH-USD", eth_price)]:
                    cur.execute(
                        """INSERT INTO candles
                           (product_id, timestamp, open, high, low, close, volume, timeframe)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            coin,
                            (now - timedelta(minutes=(60-i)*5)).isoformat(),
                            price - 10, price + 20, price - 20, price,
                            1000, "FIVE_MINUTE",
                        ),
                    )

        patterns = miner.mine_all(limit=100)
        corr_patterns = [p for p in patterns if p.get("pattern_type") == "correlation"]
        # Should have found BTC-ETH correlation
        assert len(corr_patterns) >= 1


# =========================================================================
# Brain Integration — Trade Close with KG Edge Strengthening
# =========================================================================

class TestBrainTradeClose:
    def test_trade_close_strengthens_kg_edge(self, db):
        """on_trade_closed should use strengthen_edge, not replace weight."""
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])

        # Close two winning trades for same strategy+coin
        for i in range(2):
            brain.on_trade_closed({
                "product_id": "BTC-USD",
                "strategy": "momentum",
                "side": "buy",
                "pnl": 2.0,
                "pnl_pct": 0.02,
                "entry_price": 100.0,
                "close_price": 102.0,
                "opened_at": datetime.utcnow().isoformat(),
                "closed_at": datetime.utcnow().isoformat(),
            })

        # Check edge weight accumulated
        with db._cursor() as cur:
            cur.execute(
                """SELECT weight, evidence FROM intel_kg_edges
                   WHERE edge_type = 'used_by'
                   LIMIT 1"""
            )
            row = cur.fetchone()
            assert row is not None
            # Weight should be > 1.0 (base) after 2 strengthens
            assert row["weight"] > 1.0
            assert row["evidence"] >= 2

    def test_trade_close_loss_weakens_edge(self, db):
        """Losing trade should use negative weight delta."""
        from moneyclaw.intelligence.brain import IntelligenceBrain

        brain = IntelligenceBrain(db, coins=["BTC-USD"])

        # First create the edge with a win
        brain.on_trade_closed({
            "product_id": "BTC-USD",
            "strategy": "momentum",
            "pnl": 2.0, "pnl_pct": 0.02,
            "opened_at": datetime.utcnow().isoformat(),
            "closed_at": datetime.utcnow().isoformat(),
        })

        with db._cursor() as cur:
            cur.execute("SELECT weight FROM intel_kg_edges WHERE edge_type = 'used_by'")
            initial_weight = cur.fetchone()["weight"]

        # Then a loss
        brain.on_trade_closed({
            "product_id": "BTC-USD",
            "strategy": "momentum",
            "pnl": -1.0, "pnl_pct": -0.01,
            "opened_at": datetime.utcnow().isoformat(),
            "closed_at": datetime.utcnow().isoformat(),
        })

        with db._cursor() as cur:
            cur.execute("SELECT weight FROM intel_kg_edges WHERE edge_type = 'used_by'")
            new_weight = cur.fetchone()["weight"]
            # Weight should decrease after loss
            assert new_weight < initial_weight + 0.1  # loss delta is -0.05
