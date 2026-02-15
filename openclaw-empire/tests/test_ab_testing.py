"""Test ab_testing — OpenClaw Empire."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.ab_testing import (
        ABTestEngine,
        Experiment,
        ExperimentEvent,
        ExperimentResult,
        ExperimentStatus,
        ExperimentType,
        MetricType,
        SignificanceLevel,
        Variant,
        VariantResult,
        get_engine,
        _normal_cdf,
        _normal_ppf,
        _p_to_significance,
        _load_json,
        _save_json,
        _now_iso,
        _parse_iso,
        DEFAULT_MIN_SAMPLE_SIZE,
        DEFAULT_CONFIDENCE_THRESHOLD,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="ab_testing module not available"
)


# ===================================================================
# Statistical helper functions
# ===================================================================

class TestNormalCDF:
    """Test the Abramowitz & Stegun normal CDF approximation."""

    def test_cdf_at_zero(self):
        """CDF(0) should be 0.5."""
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-6

    def test_cdf_positive(self):
        """CDF(1.96) ~ 0.975."""
        assert abs(_normal_cdf(1.96) - 0.975) < 0.002

    def test_cdf_negative(self):
        """CDF(-1.96) ~ 0.025."""
        assert abs(_normal_cdf(-1.96) - 0.025) < 0.002

    def test_cdf_extreme_positive(self):
        """CDF(8+) returns 1.0."""
        assert _normal_cdf(10.0) == 1.0

    def test_cdf_extreme_negative(self):
        """CDF(-8-) returns 0.0."""
        assert _normal_cdf(-10.0) == 0.0

    def test_cdf_symmetry(self):
        """CDF(x) + CDF(-x) should equal 1.0."""
        for x in [0.5, 1.0, 1.5, 2.0, 2.5]:
            total = _normal_cdf(x) + _normal_cdf(-x)
            assert abs(total - 1.0) < 1e-6


class TestNormalPPF:
    """Test the inverse CDF (percent point function)."""

    def test_ppf_at_half(self):
        """PPF(0.5) should be 0.0."""
        assert abs(_normal_ppf(0.5)) < 0.001

    def test_ppf_at_0975(self):
        """PPF(0.975) ~ 1.96."""
        assert abs(_normal_ppf(0.975) - 1.96) < 0.01

    def test_ppf_boundary_zero(self):
        """PPF(0) returns extreme negative."""
        assert _normal_ppf(0.0) <= -7.0

    def test_ppf_boundary_one(self):
        """PPF(1) returns extreme positive."""
        assert _normal_ppf(1.0) >= 7.0


class TestPToSignificance:
    """Test p-value to significance level mapping."""

    def test_very_high(self):
        assert _p_to_significance(0.0005) == SignificanceLevel.VERY_HIGH

    def test_high(self):
        assert _p_to_significance(0.005) == SignificanceLevel.HIGH

    def test_medium(self):
        assert _p_to_significance(0.03) == SignificanceLevel.MEDIUM

    def test_low(self):
        assert _p_to_significance(0.08) == SignificanceLevel.LOW

    def test_not_significant(self):
        assert _p_to_significance(0.15) is None


# ===================================================================
# Variant dataclass
# ===================================================================

class TestVariant:
    """Test Variant dataclass."""

    def test_default_variant(self):
        v = Variant()
        assert v.variant_id
        assert v.impressions == 0
        assert v.conversions == 0

    def test_conversion_rate_no_impressions(self):
        v = Variant(impressions=0, conversions=0)
        assert v.conversion_rate == 0.0

    def test_conversion_rate_with_data(self):
        v = Variant(impressions=1000, conversions=50)
        assert abs(v.conversion_rate - 0.05) < 1e-9

    def test_avg_value_no_conversions(self):
        v = Variant(conversions=0, total_value=100.0)
        assert v.avg_value == 0.0

    def test_avg_value_with_conversions(self):
        v = Variant(conversions=10, total_value=500.0)
        assert abs(v.avg_value - 50.0) < 1e-9

    def test_revenue_per_impression(self):
        v = Variant(impressions=2000, total_value=100.0)
        assert abs(v.revenue_per_impression - 0.05) < 1e-9

    def test_to_dict_roundtrip(self):
        v = Variant(name="Control", is_control=True, impressions=500, conversions=25)
        d = v.to_dict()
        v2 = Variant.from_dict(d)
        assert v2.name == "Control"
        assert v2.is_control is True
        assert v2.impressions == 500


# ===================================================================
# Experiment dataclass
# ===================================================================

class TestExperiment:
    """Test Experiment dataclass."""

    def test_default_experiment(self):
        exp = Experiment()
        assert exp.experiment_id
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.type == ExperimentType.HEADLINE

    def test_total_impressions(self):
        exp = Experiment(
            variants=[
                Variant(impressions=500),
                Variant(impressions=600),
            ]
        )
        assert exp.total_impressions == 1100

    def test_total_conversions(self):
        exp = Experiment(
            variants=[
                Variant(conversions=10),
                Variant(conversions=15),
            ]
        )
        assert exp.total_conversions == 25

    def test_control_returns_marked_variant(self):
        v1 = Variant(name="A", is_control=False)
        v2 = Variant(name="B", is_control=True)
        exp = Experiment(variants=[v1, v2])
        assert exp.control.name == "B"

    def test_control_defaults_to_first(self):
        v1 = Variant(name="A")
        v2 = Variant(name="B")
        exp = Experiment(variants=[v1, v2])
        assert exp.control.name == "A"

    def test_has_sufficient_data(self):
        exp = Experiment(
            min_sample_size=100,
            variants=[
                Variant(impressions=150),
                Variant(impressions=120),
            ],
        )
        assert exp.has_sufficient_data is True

    def test_insufficient_data(self):
        exp = Experiment(
            min_sample_size=100,
            variants=[
                Variant(impressions=50),
                Variant(impressions=120),
            ],
        )
        assert exp.has_sufficient_data is False

    def test_is_active(self):
        exp = Experiment(status=ExperimentStatus.RUNNING)
        assert exp.is_active is True
        exp2 = Experiment(status=ExperimentStatus.PAUSED)
        assert exp2.is_active is False

    def test_to_dict_roundtrip(self):
        exp = Experiment(
            name="CTA Test",
            type=ExperimentType.CTA,
            site_id="witchcraft",
            primary_metric=MetricType.CLICK_RATE,
            variants=[
                Variant(name="A", is_control=True, impressions=500, conversions=25),
                Variant(name="B", impressions=500, conversions=40),
            ],
        )
        d = exp.to_dict()
        assert d["type"] == "cta"
        assert d["primary_metric"] == "click_rate"
        exp2 = Experiment.from_dict(d)
        assert exp2.name == "CTA Test"
        assert len(exp2.variants) == 2
        assert exp2.variants[0].name == "A"

    def test_get_variant(self):
        v1 = Variant(variant_id="var_a", name="A")
        v2 = Variant(variant_id="var_b", name="B")
        exp = Experiment(variants=[v1, v2])
        assert exp.get_variant("var_b").name == "B"
        assert exp.get_variant("nonexistent") is None


# ===================================================================
# ABTestEngine — experiment lifecycle
# ===================================================================

class TestABTestEngineLifecycle:
    """Test experiment creation and state transitions."""

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_create_experiment(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Headline Test",
            experiment_type=ExperimentType.HEADLINE,
            site_id="witchcraft",
            page_url="https://witchcraftforbeginners.com/",
            primary_metric=MetricType.CLICK_RATE,
            variants=[
                {"name": "Control", "content": "Begin Your Journey", "is_control": True},
                {"name": "Variant B", "content": "Unlock Ancient Secrets"},
            ],
        )
        assert isinstance(exp, Experiment)
        assert exp.name == "Headline Test"
        assert exp.status == ExperimentStatus.DRAFT
        assert len(exp.variants) == 2

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_start_experiment(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Start Test",
            experiment_type=ExperimentType.HEADLINE,
            site_id="witchcraft",
            primary_metric=MetricType.CLICK_RATE,
            variants=[
                {"name": "A", "is_control": True},
                {"name": "B"},
            ],
        )
        started = await engine.start_experiment(exp.experiment_id)
        assert started.status == ExperimentStatus.RUNNING
        assert started.started_at is not None

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_pause_and_resume(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Pause Test",
            experiment_type=ExperimentType.CTA,
            site_id="witchcraft",
            primary_metric=MetricType.CONVERSION_RATE,
            variants=[{"name": "A", "is_control": True}, {"name": "B"}],
        )
        await engine.start_experiment(exp.experiment_id)
        paused = await engine.pause_experiment(exp.experiment_id)
        assert paused.status == ExperimentStatus.PAUSED

        resumed = await engine.resume_experiment(exp.experiment_id)
        assert resumed.status == ExperimentStatus.RUNNING

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_cancel_experiment(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Cancel Test",
            experiment_type=ExperimentType.IMAGE,
            site_id="witchcraft",
            primary_metric=MetricType.CLICK_RATE,
            variants=[{"name": "A", "is_control": True}, {"name": "B"}],
        )
        cancelled = await engine.cancel_experiment(exp.experiment_id)
        assert cancelled.status == ExperimentStatus.CANCELLED


# ===================================================================
# ABTestEngine — recording events
# ===================================================================

class TestABTestEngineRecording:
    """Test impression and conversion recording."""

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_record_impression(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Impression Test",
            experiment_type=ExperimentType.HEADLINE,
            site_id="witchcraft",
            primary_metric=MetricType.CLICK_RATE,
            variants=[
                {"name": "A", "is_control": True, "traffic_weight": 0.5},
                {"name": "B", "traffic_weight": 0.5},
            ],
        )
        await engine.start_experiment(exp.experiment_id)
        variant_id = await engine.record_impression(exp.experiment_id, "visitor-001")
        assert variant_id  # should return a variant ID

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_record_conversion(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Conversion Test",
            experiment_type=ExperimentType.CTA,
            site_id="witchcraft",
            primary_metric=MetricType.CONVERSION_RATE,
            variants=[{"name": "A", "is_control": True}, {"name": "B"}],
        )
        await engine.start_experiment(exp.experiment_id)
        variant_id = await engine.record_impression(exp.experiment_id, "visitor-100")
        await engine.record_conversion(exp.experiment_id, "visitor-100", value=1.0)

        # Verify counts increased
        stored = await engine.get_experiment(exp.experiment_id)
        assert stored.total_impressions >= 1


# ===================================================================
# ABTestEngine — results calculation
# ===================================================================

class TestABTestEngineResults:
    """Test statistical results calculation."""

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_calculate_results_returns_experiment_result(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Results Test",
            experiment_type=ExperimentType.HEADLINE,
            site_id="witchcraft",
            primary_metric=MetricType.CLICK_RATE,
            variants=[
                {"name": "A", "is_control": True},
                {"name": "B"},
            ],
        )
        await engine.start_experiment(exp.experiment_id)

        # Simulate bulk impressions/conversions
        for i in range(200):
            await engine.record_impression(exp.experiment_id, f"vis-a-{i}")
        for i in range(200):
            await engine.record_impression(exp.experiment_id, f"vis-b-{i}")
        # Record some conversions
        for i in range(10):
            await engine.record_conversion(exp.experiment_id, f"vis-a-{i}")
        for i in range(20):
            await engine.record_conversion(exp.experiment_id, f"vis-b-{i}")

        result = await engine.calculate_results(exp.experiment_id)
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == exp.experiment_id
        assert len(result.variant_results) == 2


# ===================================================================
# ABTestEngine — listing and filtering
# ===================================================================

class TestABTestEngineListing:
    """Test experiment listing and filtering."""

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_list_experiments(self, mock_load, mock_save):
        engine = ABTestEngine()
        await engine.create_experiment(
            name="Exp 1", experiment_type=ExperimentType.HEADLINE,
            site_id="witchcraft", primary_metric=MetricType.CLICK_RATE,
            variants=[{"name": "A", "is_control": True}, {"name": "B"}],
        )
        await engine.create_experiment(
            name="Exp 2", experiment_type=ExperimentType.CTA,
            site_id="smarthome", primary_metric=MetricType.CONVERSION_RATE,
            variants=[{"name": "A", "is_control": True}, {"name": "B"}],
        )
        all_exps = await engine.list_experiments()
        assert len(all_exps) >= 2

        site_exps = await engine.list_experiments(site_id="witchcraft")
        assert all(e.site_id == "witchcraft" for e in site_exps)

    @pytest.mark.asyncio
    @patch("src.ab_testing._save_json")
    @patch("src.ab_testing._load_json", side_effect=lambda *a, **kw: {})
    async def test_list_by_status(self, mock_load, mock_save):
        engine = ABTestEngine()
        exp = await engine.create_experiment(
            name="Status Filter",
            experiment_type=ExperimentType.HEADLINE,
            site_id="witchcraft",
            primary_metric=MetricType.CLICK_RATE,
            variants=[{"name": "A", "is_control": True}, {"name": "B"}],
        )
        await engine.start_experiment(exp.experiment_id)
        running = await engine.list_experiments(status=ExperimentStatus.RUNNING)
        assert any(e.experiment_id == exp.experiment_id for e in running)


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """Test data persistence helpers."""

    def test_save_and_load_json(self, tmp_path):
        path = tmp_path / "test.json"
        _save_json(path, {"experiments": [1, 2, 3]})
        loaded = _load_json(path)
        assert loaded == {"experiments": [1, 2, 3]}

    def test_load_missing_default(self, tmp_path):
        result = _load_json(tmp_path / "absent.json", {"empty": True})
        assert result == {"empty": True}

    def test_now_iso_format(self):
        iso = _now_iso()
        assert "T" in iso
        assert "+" in iso or "Z" in iso

    def test_parse_iso_valid(self):
        dt = _parse_iso("2026-01-15T10:30:00+00:00")
        assert dt is not None
        assert dt.year == 2026

    def test_parse_iso_invalid(self):
        assert _parse_iso("not-a-date") is None
        assert _parse_iso(None) is None
        assert _parse_iso("") is None


# ===================================================================
# ExperimentEvent dataclass
# ===================================================================

class TestExperimentEvent:
    """Test ExperimentEvent serialisation."""

    def test_default_event(self):
        ev = ExperimentEvent()
        assert ev.event_id
        assert ev.event_type == "impression"

    def test_roundtrip(self):
        ev = ExperimentEvent(
            experiment_id="exp1",
            variant_id="var1",
            event_type="conversion",
            value=5.0,
            visitor_id="vis-1",
        )
        d = ev.to_dict()
        ev2 = ExperimentEvent.from_dict(d)
        assert ev2.event_type == "conversion"
        assert ev2.value == 5.0


# ===================================================================
# ExperimentResult / VariantResult
# ===================================================================

class TestExperimentResult:
    """Test result dataclass serialisation."""

    def test_variant_result_roundtrip(self):
        vr = VariantResult(
            variant_id="v1",
            name="Control",
            is_control=True,
            impressions=1000,
            conversions=50,
            conversion_rate=0.05,
            confidence_interval=(0.038, 0.064),
            p_value=0.03,
            is_significant=True,
        )
        d = vr.to_dict()
        assert d["confidence_interval"] == [0.038, 0.064]
        vr2 = VariantResult.from_dict(d)
        assert vr2.confidence_interval == (0.038, 0.064)

    def test_experiment_result_roundtrip(self):
        er = ExperimentResult(
            experiment_id="exp1",
            winner="v2",
            confidence=0.95,
            p_value=0.02,
            lift=15.3,
            variant_results=[
                VariantResult(variant_id="v1", name="A"),
                VariantResult(variant_id="v2", name="B"),
            ],
        )
        d = er.to_dict()
        er2 = ExperimentResult.from_dict(d)
        assert er2.winner == "v2"
        assert len(er2.variant_results) == 2


# ===================================================================
# Enums
# ===================================================================

class TestEnums:
    """Verify enum values."""

    def test_experiment_types(self):
        assert ExperimentType.HEADLINE.value == "headline"
        assert ExperimentType.PRICING.value == "pricing"

    def test_metric_types(self):
        assert MetricType.CLICK_RATE.value == "click_rate"
        assert MetricType.REVENUE.value == "revenue"

    def test_significance_levels(self):
        assert SignificanceLevel.VERY_HIGH.value == "very_high"
        assert SignificanceLevel.LOW.value == "low"
