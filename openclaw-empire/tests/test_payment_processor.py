"""Test payment_processor -- OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.payment_processor import (
        PaymentProcessor,
        Payment,
        Invoice,
        InvoiceItem,
        ReconciliationRecord,
        PaymentSchedule,
        RevenueBreakdown,
        PaymentSource,
        PaymentStatus,
        InvoiceStatus,
        ReconciliationStatus,
        Currency,
        PaymentFrequency,
        get_processor,
        _load_json,
        _save_json,
        _round_amount,
        _format_currency,
        _gen_id,
        _month_bounds,
        ALL_SITE_IDS,
        SITE_DOMAIN_MAP,
        DEFAULT_CURRENCY,
        DEFAULT_PAYMENT_TERMS,
        SOURCE_FEE_RATES,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="payment_processor module not available"
)


# ===================================================================
# Enum tests
# ===================================================================

class TestEnums:
    """Verify enum values and from_string parsers."""

    def test_payment_source_values(self):
        assert PaymentSource.ADSENSE.value == "adsense"
        assert PaymentSource.KDP_ROYALTY.value == "kdp_royalty"
        assert PaymentSource.CONSULTING.value == "consulting"

    def test_payment_source_from_string(self):
        assert PaymentSource.from_string("adsense") == PaymentSource.ADSENSE
        assert PaymentSource.from_string("ADSENSE") == PaymentSource.ADSENSE
        assert PaymentSource.from_string("kdp_royalty") == PaymentSource.KDP_ROYALTY

    def test_payment_source_invalid(self):
        with pytest.raises(ValueError):
            PaymentSource.from_string("nonexistent")

    def test_payment_status_values(self):
        assert PaymentStatus.PENDING.value == "pending"
        assert PaymentStatus.RECEIVED.value == "received"
        assert PaymentStatus.REFUNDED.value == "refunded"

    def test_invoice_status_values(self):
        assert InvoiceStatus.DRAFT.value == "draft"
        assert InvoiceStatus.PAID.value == "paid"
        assert InvoiceStatus.OVERDUE.value == "overdue"

    def test_reconciliation_status(self):
        assert ReconciliationStatus.MATCHED.value == "matched"
        assert ReconciliationStatus.DISCREPANCY.value == "discrepancy"

    def test_currency(self):
        assert Currency.USD.value == "USD"
        assert Currency.from_string("eur") == Currency.EUR

    def test_payment_frequency_days(self):
        assert PaymentFrequency.DAILY.days == 1
        assert PaymentFrequency.WEEKLY.days == 7
        assert PaymentFrequency.MONTHLY.days == 30
        assert PaymentFrequency.QUARTERLY.days == 90
        assert PaymentFrequency.ONE_TIME.days == 0


# ===================================================================
# Helper functions
# ===================================================================

class TestHelpers:
    """Test module-level helper functions."""

    def test_round_amount(self):
        # Python uses banker's rounding: round(42.555, 2) == 42.55
        # Use a value that rounds unambiguously
        assert _round_amount(42.556) == 42.56
        assert _round_amount(0.1 + 0.2) == 0.3

    def test_format_currency_usd(self):
        result = _format_currency(1234.56, "USD")
        assert "$" in result
        assert "1,234.56" in result

    def test_format_currency_eur(self):
        result = _format_currency(99.99, "EUR")
        assert "99.99" in result

    def test_gen_id_prefix(self):
        pid = _gen_id("pay")
        assert pid.startswith("pay_")
        assert len(pid) > 4

    def test_month_bounds(self):
        start, end = _month_bounds(date(2026, 2, 15))
        assert start == "2026-02-01"
        assert end == "2026-02-28"

    def test_month_bounds_december(self):
        start, end = _month_bounds(date(2025, 12, 25))
        assert start == "2025-12-01"
        assert end == "2025-12-31"

    def test_constants(self):
        assert DEFAULT_CURRENCY == "USD"
        assert DEFAULT_PAYMENT_TERMS == "Net 30"
        assert len(ALL_SITE_IDS) == 16
        assert "witchcraft" in ALL_SITE_IDS

    def test_site_domain_map(self):
        assert SITE_DOMAIN_MAP["witchcraft"] == "witchcraftforbeginners.com"
        assert SITE_DOMAIN_MAP["smarthome"] == "smarthomewizards.com"

    def test_source_fee_rates(self):
        assert SOURCE_FEE_RATES["adsense"] == 0.0
        assert SOURCE_FEE_RATES["etsy_sales"] > 0


# ===================================================================
# Payment dataclass
# ===================================================================

class TestPayment:
    """Test Payment dataclass."""

    def test_auto_id_and_dates(self):
        p = Payment(source="adsense", amount=100.0)
        assert p.payment_id.startswith("pay_")
        assert p.created_at
        assert p.received_date

    def test_net_amount_auto_calc(self):
        p = Payment(source="adsense", amount=100.0, fees=5.0)
        assert p.net_amount == 95.0

    def test_from_dict(self):
        data = {
            "payment_id": "pay_abc123",
            "source": "kdp_royalty",
            "amount": 250.0,
            "status": "received",
        }
        p = Payment.from_dict(data)
        assert p.source == "kdp_royalty"
        assert p.amount == 250.0

    def test_to_dict(self):
        p = Payment(source="etsy_sales", amount=50.0)
        d = p.to_dict()
        assert d["source"] == "etsy_sales"
        assert d["amount"] == 50.0


# ===================================================================
# Invoice and InvoiceItem dataclasses
# ===================================================================

class TestInvoice:
    """Test Invoice dataclass."""

    def test_auto_id_and_dates(self):
        inv = Invoice(client_name="Test Corp")
        assert inv.invoice_id.startswith("inv_")
        assert inv.issued_date
        assert inv.due_date

    def test_recalculate(self):
        inv = Invoice(
            client_name="Client",
            items=[
                {"description": "Service A", "quantity": 1, "unit_price": 500, "total": 500},
                {"description": "Service B", "quantity": 2, "unit_price": 250, "total": 500},
            ],
            tax_rate=0.10,
        )
        assert inv.subtotal == 1000.0
        assert inv.tax_amount == 100.0
        assert inv.total == 1100.0

    def test_from_dict(self):
        data = {
            "invoice_id": "inv_test123",
            "client_name": "Acme",
            "client_email": "acme@test.com",
            "items": [],
            "status": "draft",
        }
        inv = Invoice.from_dict(data)
        assert inv.client_name == "Acme"

    def test_invoice_item(self):
        item = InvoiceItem(description="Consulting", quantity=5, unit_price=100.0)
        assert item.total == 500.0


# ===================================================================
# ReconciliationRecord
# ===================================================================

class TestReconciliationRecord:
    """Test ReconciliationRecord dataclass."""

    def test_auto_discrepancy_calc(self):
        rec = ReconciliationRecord(
            source="adsense",
            expected_amount=500.0,
            actual_amount=475.0,
        )
        assert rec.discrepancy == -25.0
        assert rec.record_id.startswith("rec_")

    def test_from_dict(self):
        data = {
            "record_id": "rec_abc",
            "source": "adsense",
            "expected_amount": 500.0,
            "actual_amount": 500.0,
            "status": "matched",
        }
        rec = ReconciliationRecord.from_dict(data)
        assert rec.status == "matched"


# ===================================================================
# PaymentSchedule
# ===================================================================

class TestPaymentSchedule:
    """Test PaymentSchedule dataclass."""

    def test_auto_next_date(self):
        sched = PaymentSchedule(
            source="adsense",
            expected_amount=150.0,
            frequency="monthly",
        )
        assert sched.schedule_id.startswith("sch_")
        assert sched.next_expected_date

    def test_from_dict(self):
        data = {
            "schedule_id": "sch_test",
            "source": "kdp_royalty",
            "expected_amount": 200.0,
            "frequency": "monthly",
        }
        sched = PaymentSchedule.from_dict(data)
        assert sched.expected_amount == 200.0


# ===================================================================
# PaymentProcessor -- payment recording
# ===================================================================

def _fresh_dict(*args, **kwargs):
    """Return a fresh empty dict for each call (avoids shared-reference bug with MagicMock return_value)."""
    return {}


class TestPaymentProcessorRecording:
    """Test payment recording on the processor."""

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_record_payment(self, mock_load, mock_save):
        proc = PaymentProcessor()
        p = proc.record_payment(
            source=PaymentSource.ADSENSE,
            amount=142.50,
            site_id="witchcraft",
            description="Feb 2026 AdSense",
        )
        assert isinstance(p, Payment)
        assert p.source == "adsense"
        assert p.amount == 142.50
        assert p.site_id == "witchcraft"

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_record_payment_string_source(self, mock_load, mock_save):
        proc = PaymentProcessor()
        p = proc.record_payment(
            source="kdp_royalty",
            amount=55.00,
        )
        assert p.source == "kdp_royalty"

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_get_payment(self, mock_load, mock_save):
        proc = PaymentProcessor()
        p = proc.record_payment(source="adsense", amount=100.0)
        fetched = proc.get_payment(p.payment_id)
        assert fetched.amount == 100.0


# ===================================================================
# Invoice generation
# ===================================================================

class TestPaymentProcessorInvoices:
    """Test invoice creation and management."""

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_create_invoice(self, mock_load, mock_save):
        proc = PaymentProcessor()
        inv = proc.create_invoice(
            client_name="Acme Corp",
            client_email="billing@acme.com",
            items=[
                {"description": "Sponsored Post", "quantity": 1, "unit_price": 500, "total": 500},
            ],
        )
        assert isinstance(inv, Invoice)
        assert inv.client_name == "Acme Corp"
        assert inv.subtotal == 500.0

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_invoice_numbering(self, mock_load, mock_save):
        proc = PaymentProcessor()
        # create_invoice requires client_email and at least one item
        inv1 = proc.create_invoice(
            client_name="A",
            client_email="a@test.com",
            items=[{"description": "Item", "quantity": 1, "unit_price": 100, "total": 100}],
        )
        inv2 = proc.create_invoice(
            client_name="B",
            client_email="b@test.com",
            items=[{"description": "Item", "quantity": 1, "unit_price": 200, "total": 200}],
        )
        # Invoice IDs should be unique
        assert inv1.invoice_id != inv2.invoice_id


# ===================================================================
# Revenue reconciliation
# ===================================================================

class TestPaymentProcessorReconciliation:
    """Test reconciliation of expected vs actual revenue."""

    @pytest.mark.asyncio
    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    async def test_reconcile(self, mock_load, mock_save):
        proc = PaymentProcessor()
        # Record some payments
        proc.record_payment(source="adsense", amount=200.0, received_date="2026-01-15")
        proc.record_payment(source="adsense", amount=250.0, received_date="2026-01-20")

        rec = await proc.reconcile(
            source="adsense",
            period_start="2026-01-01",
            period_end="2026-01-31",
            expected_amount=500.0,
        )
        assert isinstance(rec, ReconciliationRecord)
        assert rec.expected_amount == 500.0
        assert rec.actual_amount == 450.0
        assert rec.discrepancy == -50.0


# ===================================================================
# Payment search and filtering
# ===================================================================

class TestPaymentProcessorSearch:
    """Test payment listing and search."""

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_list_payments_all(self, mock_load, mock_save):
        proc = PaymentProcessor()
        proc.record_payment(source="adsense", amount=100.0)
        proc.record_payment(source="kdp_royalty", amount=200.0)
        all_payments = proc.list_payments()
        assert len(all_payments) >= 2

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_list_payments_by_source(self, mock_load, mock_save):
        proc = PaymentProcessor()
        proc.record_payment(source="adsense", amount=100.0)
        proc.record_payment(source="kdp_royalty", amount=200.0)
        adsense_payments = proc.list_payments(source="adsense")
        assert all(p.source == "adsense" for p in adsense_payments)

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_list_payments_by_status(self, mock_load, mock_save):
        proc = PaymentProcessor()
        proc.record_payment(source="adsense", amount=100.0, status="received")
        proc.record_payment(source="adsense", amount=50.0, status="pending")
        received = proc.list_payments(status="received")
        assert all(p.status == "received" for p in received)


# ===================================================================
# Revenue breakdown / report
# ===================================================================

class TestPaymentProcessorReports:
    """Test revenue breakdown and report generation."""

    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    def test_get_revenue_breakdown(self, mock_load, mock_save):
        proc = PaymentProcessor()
        proc.record_payment(source="adsense", amount=100.0, received_date="2026-01-15")
        proc.record_payment(source="kdp_royalty", amount=200.0, received_date="2026-01-20")
        proc.record_payment(source="adsense", amount=150.0, received_date="2026-01-25")

        breakdown = proc.get_revenue_breakdown("2026-01-01", "2026-01-31")
        assert isinstance(breakdown, RevenueBreakdown)
        assert breakdown.total_gross > 0
        assert breakdown.payment_count >= 3

    @pytest.mark.asyncio
    @patch("src.payment_processor._save_json")
    @patch("src.payment_processor._load_json", side_effect=_fresh_dict)
    async def test_generate_financial_report(self, mock_load, mock_save):
        proc = PaymentProcessor()
        proc.record_payment(source="adsense", amount=500.0, received_date="2026-01-15")
        report = await proc.generate_financial_report(months=1)
        assert isinstance(report, dict)


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """Test data persistence helpers."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "payments.json"
        _save_json(path, {"payments": {"pay_1": {"amount": 100}}})
        loaded = _load_json(path)
        assert "payments" in loaded

    def test_load_missing_default(self, tmp_path):
        result = _load_json(tmp_path / "absent.json", {})
        assert result == {}

    def test_load_corrupt_default(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{{invalid", encoding="utf-8")
        result = _load_json(path, {"fallback": True})
        assert result == {"fallback": True}
