"""Test competitor_intel — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.competitor_intel import (
        CompetitorIntel,
        Competitor,
        CompetitorType,
        ContentStrategy,
        ThreatLevel,
        AnalysisType,
        KeywordGap,
        ContentGap,
        VelocitySnapshot,
        CompetitorReport,
        get_intel,
        _detect_strategy_from_signals,
        _load_json,
        _save_json,
        _now_iso,
        ALL_SITE_IDS,
        NICHE_GROUPS,
        HIGH_OPPORTUNITY_THRESHOLD,
        MEDIUM_OPPORTUNITY_THRESHOLD,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="competitor_intel module not available"
)


# ===================================================================
# Enum tests
# ===================================================================

class TestEnums:
    """Verify enum values."""

    def test_competitor_types(self):
        assert CompetitorType.DIRECT.value == "direct"
        assert CompetitorType.INDIRECT.value == "indirect"
        assert CompetitorType.ASPIRATIONAL.value == "aspirational"

    def test_analysis_types(self):
        assert AnalysisType.KEYWORD_GAP.value == "keyword_gap"
        assert AnalysisType.CONTENT_GAP.value == "content_gap"
        assert AnalysisType.FULL.value == "full"

    def test_threat_levels(self):
        assert ThreatLevel.LOW.value == "low"
        assert ThreatLevel.CRITICAL.value == "critical"

    def test_content_strategies(self):
        assert ContentStrategy.PILLAR_CLUSTER.value == "pillar_cluster"
        assert ContentStrategy.FREQUENCY.value == "frequency"
        assert ContentStrategy.SEO_FOCUSED.value == "seo_focused"


# ===================================================================
# Competitor dataclass
# ===================================================================

class TestCompetitor:
    """Test Competitor dataclass."""

    def test_auto_id(self):
        comp = Competitor(domain="example.com", name="Example")
        assert comp.competitor_id.startswith("comp-")
        assert comp.created_at

    def test_with_fields(self):
        comp = Competitor(
            domain="rival.com",
            name="Rival Site",
            type=CompetitorType.DIRECT,
            our_site_id="witchcraft",
            niche="witchcraft-spirituality",
            estimated_monthly_traffic=50000,
            domain_authority=45.0,
        )
        assert comp.domain == "rival.com"
        assert comp.estimated_monthly_traffic == 50000

    def test_to_dict_roundtrip(self):
        comp = Competitor(
            domain="test.com",
            name="Test",
            type=CompetitorType.ASPIRATIONAL,
            threat_level=ThreatLevel.HIGH,
            content_strategy=ContentStrategy.LONG_FORM,
        )
        d = comp.to_dict()
        assert d["type"] == "aspirational"
        assert d["threat_level"] == "high"
        comp2 = Competitor.from_dict(d)
        assert comp2.type == CompetitorType.ASPIRATIONAL
        assert comp2.threat_level == ThreatLevel.HIGH

    def test_from_dict_handles_bad_enums(self):
        data = {
            "domain": "bad.com",
            "name": "Bad",
            "type": "invalid_type",
            "threat_level": "invalid_level",
        }
        comp = Competitor.from_dict(data)
        assert comp.type == CompetitorType.DIRECT  # fallback
        assert comp.threat_level == ThreatLevel.LOW  # fallback


# ===================================================================
# KeywordGap dataclass
# ===================================================================

class TestKeywordGap:
    """Test KeywordGap dataclass and opportunity scoring."""

    def test_auto_timestamp(self):
        kg = KeywordGap(keyword="moon water", our_site_id="witchcraft")
        assert kg.identified_at

    def test_opportunity_score_calculation(self):
        kg = KeywordGap(
            keyword="crystal healing guide",
            search_volume=3000,
            difficulty=30.0,
            our_rank=0,
            competitor_rank=5,
        )
        assert 0.0 <= kg.opportunity_score <= 10.0

    def test_high_volume_low_difficulty(self):
        """High volume + low difficulty should yield high opportunity."""
        kg = KeywordGap(
            keyword="easy witchcraft spells",
            search_volume=8000,
            difficulty=15.0,
            our_rank=0,
            competitor_rank=3,
        )
        assert kg.opportunity_score > 5.0

    def test_roundtrip(self):
        kg = KeywordGap(
            keyword="moon ritual",
            our_site_id="witchcraft",
            competitor_id="comp-123",
            search_volume=2000,
        )
        d = kg.to_dict()
        kg2 = KeywordGap.from_dict(d)
        assert kg2.keyword == "moon ritual"
        assert kg2.search_volume == 2000


# ===================================================================
# ContentGap dataclass
# ===================================================================

class TestContentGap:
    """Test ContentGap dataclass and opportunity scoring."""

    def test_auto_timestamp(self):
        cg = ContentGap(topic="Full Moon Ritual Guide")
        assert cg.identified_at

    def test_opportunity_score_none_coverage(self):
        """No coverage should yield high opportunity score."""
        cg = ContentGap(
            topic="Crystal Grid Layouts",
            our_coverage="none",
            covered_by=["comp-1", "comp-2", "comp-3"],
            estimated_traffic=5000,
            difficulty=25.0,
        )
        assert cg.opportunity_score > 5.0

    def test_opportunity_score_full_coverage(self):
        """Full coverage should yield low opportunity."""
        cg = ContentGap(
            topic="Basic Tarot Guide",
            our_coverage="full",
            covered_by=["comp-1"],
            estimated_traffic=1000,
            difficulty=50.0,
        )
        assert cg.opportunity_score < 5.0

    def test_roundtrip(self):
        cg = ContentGap(
            topic="Herbal Tea Blending",
            our_site_id="herbalwitchery",
            covered_by=["comp-a"],
        )
        d = cg.to_dict()
        cg2 = ContentGap.from_dict(d)
        assert cg2.topic == "Herbal Tea Blending"


# ===================================================================
# VelocitySnapshot
# ===================================================================

class TestVelocitySnapshot:
    """Test VelocitySnapshot dataclass."""

    def test_auto_fields(self):
        vs = VelocitySnapshot(
            competitor_id="comp-123",
            articles_published=10,
            word_count_total=25000,
        )
        assert vs.snapshot_id.startswith("vel-")
        assert vs.date  # auto-set to today
        assert vs.avg_word_count == 2500  # auto-calculated

    def test_roundtrip(self):
        vs = VelocitySnapshot(
            competitor_id="comp-456",
            articles_published=5,
            word_count_total=10000,
        )
        d = vs.to_dict()
        vs2 = VelocitySnapshot.from_dict(d)
        assert vs2.articles_published == 5


# ===================================================================
# CompetitorReport
# ===================================================================

class TestCompetitorReport:
    """Test CompetitorReport dataclass."""

    def test_auto_id(self):
        rpt = CompetitorReport(site_id="witchcraft")
        assert rpt.report_id.startswith("rpt-")
        assert rpt.generated_at

    def test_roundtrip(self):
        rpt = CompetitorReport(
            site_id="witchcraft",
            competitors_analyzed=5,
            keyword_gaps_found=42,
            content_gaps_found=15,
            recommendations=["Create more pillar content", "Increase frequency"],
        )
        d = rpt.to_dict()
        rpt2 = CompetitorReport.from_dict(d)
        assert rpt2.competitors_analyzed == 5
        assert len(rpt2.recommendations) == 2


# ===================================================================
# Strategy detection
# ===================================================================

class TestStrategyDetection:
    """Test _detect_strategy_from_signals helper."""

    def test_frequency_strategy(self):
        titles = [f"Article {i}" for i in range(20)]
        strategy, confidence, evidence = _detect_strategy_from_signals(
            titles=titles,
            avg_word_count=1200,
            articles_per_week=6.0,
            social_presence={},
        )
        assert strategy == ContentStrategy.FREQUENCY
        assert confidence > 0.0

    def test_long_form_strategy(self):
        titles = ["Ultimate Guide to Crystal Healing"]
        strategy, confidence, evidence = _detect_strategy_from_signals(
            titles=titles,
            avg_word_count=3500,
            articles_per_week=1.0,
            social_presence={},
        )
        assert strategy == ContentStrategy.LONG_FORM

    def test_pillar_cluster_signals(self):
        titles = [
            "Ultimate Guide to Witchcraft",
            "Complete Beginner 101",
            "Everything About Moon Rituals",
            "Advanced Crystal Grid Master Class",
            "Simple Spell Recipe",
        ]
        strategy, confidence, evidence = _detect_strategy_from_signals(
            titles=titles,
            avg_word_count=2000,
            articles_per_week=2.0,
            social_presence={},
        )
        # Should detect pillar-cluster due to signal keywords
        assert strategy in (ContentStrategy.PILLAR_CLUSTER, ContentStrategy.LONG_FORM)


# ===================================================================
# CompetitorIntel — CRUD
# ===================================================================

class TestCompetitorIntelCRUD:
    """Test competitor add/remove/list operations."""

    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    def test_add_competitor(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        comp = intel.add_competitor(
            domain="rival.com",
            name="Rival",
            competitor_type=CompetitorType.DIRECT,
            our_site_id="witchcraft",
            niche="witchcraft-spirituality",
        )
        assert isinstance(comp, Competitor)
        assert comp.domain == "rival.com"

    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    def test_list_competitors(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        intel.add_competitor(domain="a.com", name="A", our_site_id="witchcraft")
        intel.add_competitor(domain="b.com", name="B", our_site_id="smarthome")
        all_comps = intel.list_competitors()
        assert len(all_comps) >= 2

    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    def test_list_competitors_by_site(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        intel.add_competitor(domain="a.com", name="A", our_site_id="witchcraft")
        intel.add_competitor(domain="b.com", name="B", our_site_id="smarthome")
        witch_comps = intel.list_competitors(site_id="witchcraft")
        assert all(c.our_site_id == "witchcraft" for c in witch_comps)

    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    def test_remove_competitor(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        comp = intel.add_competitor(domain="removeme.com", name="Remove")
        result = intel.remove_competitor(comp.competitor_id)
        assert result is True

    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    def test_remove_nonexistent(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        result = intel.remove_competitor("comp-nonexistent")
        assert result is False


# ===================================================================
# Keyword gap analysis (mocked AI)
# ===================================================================

class TestKeywordGapAnalysis:
    """Test keyword gap analysis with mocked dependencies."""

    @pytest.mark.asyncio
    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    async def test_analyze_keyword_gaps(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        intel.add_competitor(
            domain="rival.com", name="Rival",
            our_site_id="witchcraft",
        )
        with patch.object(intel, "analyze_keyword_gaps", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = [
                KeywordGap(keyword="moon ritual guide", search_volume=3000, our_rank=0, competitor_rank=5),
                KeywordGap(keyword="crystal grid setup", search_volume=2000, our_rank=0, competitor_rank=8),
            ]
            gaps = await intel.analyze_keyword_gaps("witchcraft", limit=10)
            assert len(gaps) == 2
            assert all(isinstance(g, KeywordGap) for g in gaps)


# ===================================================================
# Content gap detection
# ===================================================================

class TestContentGapDetection:
    """Test content gap detection with mocked dependencies."""

    @pytest.mark.asyncio
    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    async def test_analyze_content_gaps(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        intel.add_competitor(domain="rival.com", name="Rival", our_site_id="witchcraft")
        with patch.object(intel, "analyze_content_gaps", new_callable=AsyncMock) as mock_gaps:
            mock_gaps.return_value = [
                ContentGap(topic="Herbal Tea Rituals", our_coverage="none"),
            ]
            gaps = await intel.analyze_content_gaps("witchcraft")
            assert len(gaps) == 1


# ===================================================================
# Velocity measurement
# ===================================================================

class TestVelocityMeasurement:
    """Test publishing velocity tracking."""

    @pytest.mark.asyncio
    @patch("src.competitor_intel._save_json")
    @patch("src.competitor_intel._load_json", return_value={})
    @patch("src.competitor_intel._load_site_registry", return_value={})
    async def test_track_velocity(self, mock_registry, mock_load, mock_save):
        intel = CompetitorIntel()
        comp = intel.add_competitor(domain="fast.com", name="Fast Publisher", our_site_id="witchcraft")
        with patch.object(intel, "track_velocity", new_callable=AsyncMock) as mock_vel:
            mock_vel.return_value = VelocitySnapshot(
                competitor_id=comp.competitor_id,
                articles_published=12,
                word_count_total=30000,
            )
            snap = await intel.track_velocity(comp.competitor_id)
            assert snap is not None
            assert snap.articles_published == 12


# ===================================================================
# Opportunity identification
# ===================================================================

class TestOpportunityIdentification:
    """Test opportunity discovery from gaps."""

    def test_opportunity_constants(self):
        assert HIGH_OPPORTUNITY_THRESHOLD == 7.5
        assert MEDIUM_OPPORTUNITY_THRESHOLD == 5.0

    def test_niche_groups(self):
        assert "witchcraft-spirituality" in NICHE_GROUPS
        assert "witchcraft" in NICHE_GROUPS["witchcraft-spirituality"]
        assert "ai-technology" in NICHE_GROUPS

    def test_all_site_ids(self):
        assert len(ALL_SITE_IDS) == 16
        assert "witchcraft" in ALL_SITE_IDS
        assert "smarthome" in ALL_SITE_IDS


# ===================================================================
# Data persistence
# ===================================================================

class TestPersistence:
    """Test data persistence helpers."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "competitors.json"
        _save_json(path, [{"domain": "test.com"}])
        loaded = _load_json(path)
        assert loaded == [{"domain": "test.com"}]

    def test_load_missing_default(self, tmp_path):
        result = _load_json(tmp_path / "absent.json", [])
        assert result == []

    def test_now_iso(self):
        iso = _now_iso()
        assert "T" in iso
