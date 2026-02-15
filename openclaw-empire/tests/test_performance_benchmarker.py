"""Test performance_benchmarker — OpenClaw Empire."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    """Redirect performance data to temp dir."""
    perf_dir = tmp_path / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    (perf_dir / "daily").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.performance_benchmarker.PERF_DATA_DIR", perf_dir)
    monkeypatch.setattr("src.performance_benchmarker.DAILY_DIR", perf_dir / "daily")
    monkeypatch.setattr("src.performance_benchmarker.SLAS_FILE", perf_dir / "slas.json")
    monkeypatch.setattr("src.performance_benchmarker.MEASUREMENTS_FILE", perf_dir / "measurements.json")
    monkeypatch.setattr("src.performance_benchmarker.CONFIG_FILE", perf_dir / "config.json")
    # Reset singleton
    import src.performance_benchmarker as pb_mod
    pb_mod._benchmarker = None
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.performance_benchmarker import (
    COMPARISON_OPS,
    Measurement,
    MetricType,
    OperationTimer,
    PercentileReport,
    PerformanceBenchmarker,
    SLADefinition,
    SLAReport,
    SLAStatus,
    benchmark,
    get_benchmarker,
    _percentile,
    _std_dev,
)


# ===================================================================
# Enum tests
# ===================================================================

class TestMetricTypeEnum:
    def test_values(self):
        assert MetricType.LATENCY.value == "latency"
        assert MetricType.THROUGHPUT.value == "throughput"
        assert MetricType.ERROR_RATE.value == "error_rate"
        assert MetricType.TOKEN_USAGE.value == "token_usage"
        assert MetricType.COST.value == "cost"
        assert MetricType.QUEUE_DEPTH.value == "queue_depth"


class TestSLAStatusEnum:
    def test_values(self):
        assert SLAStatus.MET.value == "met"
        assert SLAStatus.WARNING.value == "warning"
        assert SLAStatus.VIOLATED.value == "violated"
        assert SLAStatus.NOT_CONFIGURED.value == "not_configured"


# ===================================================================
# Pure-Python math helpers
# ===================================================================

class TestPercentileMath:
    def test_empty_list_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_median_of_even_list(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert _percentile(values, 50) == pytest.approx(2.5, abs=0.01)

    def test_p95_of_100_values(self):
        values = list(range(1, 101))
        float_vals = [float(v) for v in values]
        p95 = _percentile(float_vals, 95)
        assert p95 == pytest.approx(95.05, abs=0.1)

    def test_p99(self):
        values = [float(i) for i in range(1, 1001)]
        p99 = _percentile(values, 99)
        assert p99 > 990

    def test_std_dev(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        mean = sum(values) / len(values)
        sd = _std_dev(values, mean)
        assert sd == pytest.approx(2.0, abs=0.1)

    def test_std_dev_single_value(self):
        assert _std_dev([5.0], 5.0) == 0.0


# ===================================================================
# Measurement
# ===================================================================

class TestMeasurement:
    def test_to_dict_and_from_dict_roundtrip(self):
        m = Measurement(
            timestamp="2026-02-15T10:00:00+00:00",
            module="wordpress_client",
            operation="publish_post",
            metric_type=MetricType.LATENCY.value,
            value=1230.5,
            unit="ms",
            metadata={"site_id": "witchcraft"},
        )
        d = m.to_dict()
        assert d["module"] == "wordpress_client"
        assert d["value"] == 1230.5

        restored = Measurement.from_dict(d)
        assert restored.module == "wordpress_client"
        assert restored.value == 1230.5


# ===================================================================
# SLADefinition
# ===================================================================

class TestSLADefinition:
    def test_to_dict_and_from_dict_roundtrip(self):
        sla = SLADefinition(
            sla_id="sla-test",
            module="api",
            operation="call",
            metric_type=MetricType.LATENCY.value,
            threshold=5000.0,
            comparison="lt",
            window_minutes=60,
            description="API latency < 5s",
        )
        d = sla.to_dict()
        restored = SLADefinition.from_dict(d)
        assert restored.sla_id == "sla-test"
        assert restored.threshold == 5000.0
        assert restored.comparison == "lt"


# ===================================================================
# PerformanceBenchmarker — Recording
# ===================================================================

class TestRecording:
    """Test metric recording."""

    @pytest.mark.asyncio
    async def test_record_measurement(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        m = await bench.record(
            "wordpress_client", "publish_post",
            MetricType.LATENCY, 1200.5, "ms",
        )
        assert m.module == "wordpress_client"
        assert m.operation == "publish_post"
        assert m.value == 1200.5

    def test_record_sync(self):
        bench = PerformanceBenchmarker()
        bench.initialize_sync()
        m = bench.record_sync(
            "wp", "get_post",
            MetricType.LATENCY, 500.0, "ms",
        )
        assert m.value == 500.0

    @pytest.mark.asyncio
    async def test_record_multiple_metrics(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        await bench.record("mod", "op", MetricType.LATENCY, 100.0, "ms")
        await bench.record("mod", "op", MetricType.LATENCY, 200.0, "ms")
        await bench.record("mod", "op", MetricType.LATENCY, 300.0, "ms")
        key = bench._make_key("mod", "op")
        assert len(bench._measurements[key]) == 3

    @pytest.mark.asyncio
    async def test_auto_unit_detection(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        m = await bench.record("mod", "op", MetricType.COST, 0.05)
        assert m.unit == "$"


# ===================================================================
# Percentile reporting
# ===================================================================

class TestPercentileReporting:
    @pytest.mark.asyncio
    async def test_get_percentiles_empty(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        report = await bench.get_percentiles("mod", "op", MetricType.LATENCY, "1h")
        assert report.count == 0
        assert report.p95 == 0.0

    @pytest.mark.asyncio
    async def test_get_percentiles_with_data(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        for v in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            await bench.record("wp", "post", MetricType.LATENCY, float(v), "ms")
        report = await bench.get_percentiles("wp", "post", MetricType.LATENCY, "1h")
        assert report.count == 10
        assert report.min == 100.0
        assert report.max == 1000.0
        assert report.mean == pytest.approx(550.0, abs=1.0)
        assert report.p95 > 0
        assert report.p99 > report.p95

    @pytest.mark.asyncio
    async def test_get_percentiles_returns_report_object(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        await bench.record("m", "o", MetricType.LATENCY, 42.0, "ms")
        report = await bench.get_percentiles("m", "o")
        assert isinstance(report, PercentileReport)
        assert report.module == "m"
        assert report.operation == "o"


# ===================================================================
# @benchmark decorator
# ===================================================================

class TestBenchmarkDecorator:
    """Test the @benchmark decorator for auto-latency recording."""

    @pytest.mark.asyncio
    async def test_async_decorator_records_latency(self):
        bench = PerformanceBenchmarker()
        bench.initialize_sync()
        with patch("src.performance_benchmarker.get_benchmarker", return_value=bench):
            @benchmark("test_mod", "test_op")
            async def my_async_func(x: int) -> int:
                return x * 2

            result = await my_async_func(5)
            assert result == 10

        key = bench._make_key("test_mod", "test_op")
        assert key in bench._measurements
        assert len(bench._measurements[key]) == 1
        m = bench._measurements[key][0]
        assert m.metric_type == MetricType.LATENCY.value
        assert m.value > 0
        assert m.metadata["success"] is True

    def test_sync_decorator_records_latency(self):
        bench = PerformanceBenchmarker()
        bench.initialize_sync()
        with patch("src.performance_benchmarker.get_benchmarker", return_value=bench):
            @benchmark("sync_mod", "sync_op")
            def my_sync_func(x: int) -> int:
                return x + 1

            result = my_sync_func(10)
            assert result == 11

        key = bench._make_key("sync_mod", "sync_op")
        assert key in bench._measurements
        m = bench._measurements[key][0]
        assert m.metadata["success"] is True

    @pytest.mark.asyncio
    async def test_decorator_captures_failure(self):
        bench = PerformanceBenchmarker()
        bench.initialize_sync()
        with patch("src.performance_benchmarker.get_benchmarker", return_value=bench):
            @benchmark("fail_mod", "fail_op")
            async def failing_func():
                raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await failing_func()

        key = bench._make_key("fail_mod", "fail_op")
        m = bench._measurements[key][0]
        assert m.metadata["success"] is False
        assert m.metadata["error_type"] == "ValueError"


# ===================================================================
# SLA monitoring
# ===================================================================

class TestSLAMonitoring:
    """Test SLA definition, evaluation, and violation detection."""

    @pytest.mark.asyncio
    async def test_define_sla(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        sla = await bench.define_sla(
            module="wp", operation="post",
            metric_type=MetricType.LATENCY,
            threshold=5000.0, comparison="lt",
            description="WP posts < 5s",
        )
        assert sla.threshold == 5000.0
        assert sla.comparison == "lt"

    @pytest.mark.asyncio
    async def test_check_sla_met(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        sla = await bench.define_sla(
            module="wp", operation="post",
            metric_type=MetricType.LATENCY,
            threshold=5000.0, comparison="lt",
        )
        # Record a measurement well under threshold
        await bench.record("wp", "post", MetricType.LATENCY, 1000.0, "ms")
        report = await bench.check_sla(sla.sla_id)
        assert report.status == SLAStatus.MET

    @pytest.mark.asyncio
    async def test_check_sla_violated(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        sla = await bench.define_sla(
            module="wp", operation="post",
            metric_type=MetricType.LATENCY,
            threshold=1000.0, comparison="lt",
        )
        # Record measurements above threshold
        await bench.record("wp", "post", MetricType.LATENCY, 5000.0, "ms")
        await bench.record("wp", "post", MetricType.LATENCY, 6000.0, "ms")
        report = await bench.check_sla(sla.sla_id)
        assert report.status == SLAStatus.VIOLATED

    @pytest.mark.asyncio
    async def test_check_sla_not_configured(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        report = await bench.check_sla("nonexistent-sla-id")
        assert report.status == SLAStatus.NOT_CONFIGURED

    @pytest.mark.asyncio
    async def test_get_violations(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        await bench.define_sla(
            module="slow", operation="op",
            metric_type=MetricType.LATENCY,
            threshold=100.0, comparison="lt",
        )
        await bench.record("slow", "op", MetricType.LATENCY, 500.0, "ms")
        violations = await bench.get_violations()
        assert len(violations) >= 1
        assert any(v.status == SLAStatus.VIOLATED for v in violations)

    @pytest.mark.asyncio
    async def test_remove_sla(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        sla = await bench.define_sla(
            module="rm", operation="op",
            metric_type=MetricType.LATENCY,
            threshold=100.0, comparison="lt",
        )
        result = await bench.remove_sla(sla.sla_id)
        assert result is True
        result2 = await bench.remove_sla(sla.sla_id)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_invalid_comparison_raises(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        with pytest.raises(ValueError, match="Invalid comparison"):
            await bench.define_sla(
                module="x", operation="y",
                metric_type=MetricType.LATENCY,
                threshold=100.0, comparison="invalid",
            )


# ===================================================================
# Module reports
# ===================================================================

class TestModuleReports:
    @pytest.mark.asyncio
    async def test_get_module_report(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        await bench.record("wp", "post", MetricType.LATENCY, 100.0, "ms")
        await bench.record("wp", "post", MetricType.LATENCY, 200.0, "ms")
        await bench.record("wp", "media", MetricType.LATENCY, 500.0, "ms")
        report = await bench.get_module_report("wp", period="1h")
        assert report["module"] == "wp"
        assert "post" in report["operations"]
        assert "media" in report["operations"]


# ===================================================================
# Statistics
# ===================================================================

class TestStats:
    @pytest.mark.asyncio
    async def test_get_stats(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        await bench.record("mod", "op", MetricType.LATENCY, 42.0)
        stats = await bench.get_stats()
        assert stats["total_measurements_in_memory"] >= 1
        assert stats["modules_count"] >= 1
        assert "mod" in stats["modules_tracked"]
        assert stats["slas_defined"] >= 0

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        stats = await bench.get_stats()
        assert stats["total_measurements_in_memory"] == 0


# ===================================================================
# OperationTimer
# ===================================================================

class TestOperationTimer:
    @pytest.mark.asyncio
    async def test_async_timer_records_latency(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        async with bench.time("timer_mod", "timer_op") as timer:
            await asyncio.sleep(0.01)
        assert timer.elapsed_ms > 0
        key = bench._make_key("timer_mod", "timer_op")
        assert key in bench._measurements

    @pytest.mark.asyncio
    async def test_async_timer_captures_error(self):
        bench = PerformanceBenchmarker()
        await bench.initialize()
        with pytest.raises(RuntimeError, match="test error"):
            async with bench.time("err_mod", "err_op") as timer:
                raise RuntimeError("test error")
        key = bench._make_key("err_mod", "err_op")
        m = bench._measurements[key][0]
        assert m.metadata["success"] is False
        assert m.metadata["error_type"] == "RuntimeError"


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_benchmarker_returns_same_instance(self):
        b1 = get_benchmarker()
        b2 = get_benchmarker()
        assert b1 is b2

    def test_get_benchmarker_is_initialized(self):
        b = get_benchmarker()
        assert b._initialized is True
