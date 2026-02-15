"""Test ocr_extractor — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.ocr_extractor import (
        ExtractionResult,
        ExtractionSchedule,
        ExtractionTemplate,
        AnomalyFlag,
        TrendPoint,
        TrendAnalysis,
        PeriodComparison,
        parse_number,
        parse_currency,
        parse_percentage,
        parse_date_range,
        parse_table,
        clean_ocr_text,
        EXTRACTION_TYPES,
        CONFIDENCE_THRESHOLD,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="ocr_extractor not available")


# ===================================================================
# Constants Tests
# ===================================================================


class TestConstants:
    def test_extraction_types(self):
        expected = [
            "adsense", "analytics", "search_console", "amazon_associates",
            "etsy", "kdp", "wordpress", "instagram", "tiktok", "pinterest",
            "twitter", "generic",
        ]
        assert EXTRACTION_TYPES == expected

    def test_confidence_threshold(self):
        assert 0 < CONFIDENCE_THRESHOLD < 1


# ===================================================================
# Number / Text Parsing Tests
# ===================================================================


class TestParseNumber:
    def test_simple_integer(self):
        assert parse_number("1234") == 1234.0

    def test_float(self):
        assert parse_number("12.34") == 12.34

    def test_with_commas(self):
        assert parse_number("1,234,567") == 1234567.0

    def test_with_k_suffix(self):
        assert parse_number("1.5K") == 1500.0

    def test_with_m_suffix(self):
        assert parse_number("2.3M") == 2300000.0

    def test_with_b_suffix(self):
        assert parse_number("1B") == 1_000_000_000.0

    def test_with_dollar_sign(self):
        assert parse_number("$1,234.56") == 1234.56

    def test_with_percentage(self):
        assert parse_number("45.5%") == 45.5

    def test_negative_parenthetical(self):
        assert parse_number("(123.45)") == -123.45

    def test_negative_dash(self):
        assert parse_number("-42") == -42.0

    def test_none_input(self):
        assert parse_number(None) is None

    def test_empty_string(self):
        assert parse_number("") is None

    def test_invalid_string(self):
        assert parse_number("abc") is None

    def test_whitespace(self):
        assert parse_number("  1234  ") == 1234.0


class TestParseCurrency:
    def test_dollar_format(self):
        assert parse_currency("$1,234.56") == 1234.56

    def test_usd_suffix(self):
        assert parse_currency("1234.56 USD") == 1234.56

    def test_usd_prefix(self):
        assert parse_currency("USD 1234.56") == 1234.56

    def test_none_input(self):
        assert parse_currency(None) is None

    def test_empty_string(self):
        assert parse_currency("") is None


class TestParsePercentage:
    def test_simple(self):
        assert parse_percentage("12.5%") == 12.5

    def test_with_space(self):
        assert parse_percentage("12.5 %") == 12.5

    def test_word_percent(self):
        assert parse_percentage("12.5 percent") == 12.5

    def test_none_input(self):
        assert parse_percentage(None) is None

    def test_empty_string(self):
        assert parse_percentage("") is None


class TestParseDateRange:
    def test_last_n_days(self):
        result = parse_date_range("Last 7 days")
        assert result is not None
        assert "start" in result
        assert "end" in result

    def test_last_n_months(self):
        result = parse_date_range("Last 3 months")
        assert result is not None

    def test_iso_date_range(self):
        result = parse_date_range("2026-01-01 to 2026-01-31")
        assert result == {"start": "2026-01-01", "end": "2026-01-31"}

    def test_none_input(self):
        assert parse_date_range(None) is None

    def test_empty_string(self):
        assert parse_date_range("") is None


class TestParseTable:
    def test_tab_separated(self):
        text = "Homepage\t1000\t50\nAbout Us\t2000\t100"
        columns = ["URL", "Views", "Clicks"]
        rows = parse_table(text, columns)
        assert len(rows) == 2
        assert rows[0]["URL"] == "Homepage"
        assert rows[0]["Views"] == "1000"

    def test_space_separated(self):
        text = "Homepage    1000    50\nAbout Us    2000    100"
        columns = ["URL", "Views", "Clicks"]
        rows = parse_table(text, columns)
        assert len(rows) == 2

    def test_empty_input(self):
        assert parse_table("", ["A", "B"]) == []

    def test_empty_columns(self):
        assert parse_table("data here", []) == []

    def test_skips_separator_lines(self):
        text = "---\nHomepage\t100\n==="
        columns = ["URL", "Value"]
        rows = parse_table(text, columns)
        assert len(rows) == 1


class TestCleanOcrText:
    def test_en_dash(self):
        assert "-" in clean_ocr_text("2020\u20132025")

    def test_em_dash(self):
        assert "-" in clean_ocr_text("hello\u2014world")

    def test_pipe_before_digit(self):
        assert clean_ocr_text("|23") == "123"

    def test_o_between_digits(self):
        assert clean_ocr_text("1O3") == "103"

    def test_empty_input(self):
        assert clean_ocr_text("") == ""

    def test_preserves_normal_text(self):
        assert clean_ocr_text("Hello World") == "Hello World"


# ===================================================================
# Data Class Tests
# ===================================================================


class TestExtractionResult:
    def test_defaults(self):
        er = ExtractionResult(
            extraction_id="test",
            source_app="Google AdSense",
            extraction_type="adsense",
        )
        assert er.confidence == 0.0
        assert er.errors == []
        assert er.timestamp != ""

    def test_auto_generates_id(self):
        er = ExtractionResult(
            extraction_id="",
            source_app="Test",
            extraction_type="generic",
        )
        assert er.extraction_id != ""

    def test_to_dict(self):
        er = ExtractionResult(
            extraction_id="ext-1",
            source_app="Etsy",
            extraction_type="etsy",
            structured_data={"revenue": 150.00},
            confidence=0.95,
        )
        d = er.to_dict()
        assert d["source_app"] == "Etsy"
        assert d["confidence"] == 0.95

    def test_from_dict(self):
        data = {
            "extraction_id": "ext-2",
            "source_app": "KDP",
            "extraction_type": "kdp",
            "confidence": 0.88,
        }
        er = ExtractionResult.from_dict(data)
        assert er.source_app == "KDP"


class TestExtractionSchedule:
    def test_defaults(self):
        es = ExtractionSchedule(
            schedule_id="sched-1",
            app_name="Google Analytics",
            extraction_type="analytics",
        )
        assert es.enabled is True
        assert es.results_count == 0


class TestAnomalyFlag:
    def test_to_dict(self):
        af = AnomalyFlag(
            metric_path="earnings.today",
            current_value=150.0,
            historical_avg=100.0,
            deviation_pct=50.0,
            severity="warning",
        )
        d = af.to_dict()
        assert d["deviation_pct"] == 50.0
        assert d["severity"] == "warning"


class TestTrendAnalysis:
    def test_defaults(self):
        ta = TrendAnalysis(
            app_name="AdSense",
            metric_path="earnings.today",
            period_days=30,
        )
        assert ta.trend_direction == "stable"
        assert ta.data_points == []


# ===================================================================
# ExtractionTemplate Tests
# ===================================================================


class TestExtractionTemplate:
    def test_get_known_type(self):
        template = ExtractionTemplate.get("adsense")
        assert template["name"] == "Google AdSense"
        assert "system_prompt" in template
        assert "expected_fields" in template

    def test_get_unknown_falls_back_to_generic(self):
        template = ExtractionTemplate.get("unknown_platform")
        assert template["name"] == "Generic Dashboard"

    def test_list_types(self):
        types = ExtractionTemplate.list_types()
        assert len(types) >= 12
        names = [t["type"] for t in types]
        assert "adsense" in names
        assert "generic" in names

    def test_all_extraction_types_have_templates(self):
        for ext_type in EXTRACTION_TYPES:
            template = ExtractionTemplate.get(ext_type)
            assert "system_prompt" in template
            assert "expected_fields" in template

    def test_adsense_expected_fields(self):
        template = ExtractionTemplate.get("adsense")
        assert "earnings.today" in template["expected_fields"]

    def test_analytics_expected_fields(self):
        template = ExtractionTemplate.get("analytics")
        assert "sessions" in template["expected_fields"]

    def test_etsy_expected_fields(self):
        template = ExtractionTemplate.get("etsy")
        assert "revenue" in template["expected_fields"]
