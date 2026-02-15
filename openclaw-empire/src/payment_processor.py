"""
Payment Processor — OpenClaw Empire Edition
============================================

Comprehensive payment tracking, invoice generation, and revenue reconciliation
across all income streams for Nick Creighton's 16-site WordPress publishing
empire. Handles AdSense, Amazon Affiliates, KDP royalties, Etsy sales,
Substack subscriptions, sponsorships, consulting, digital products, and courses.

Capabilities:
    - Record and track payments from 10+ revenue sources
    - Generate professional HTML invoices with automatic numbering
    - Reconcile expected vs actual revenue per source per period
    - Track payment schedules and flag overdue income
    - Revenue analytics: breakdowns, trends, fee analysis, velocity
    - Tax summaries by year with categorized income
    - Financial reporting with multi-month aggregation

Data storage: data/payments/
    payments.json        -- All payment records
    invoices.json        -- Invoice registry
    reconciliation.json  -- Reconciliation history
    config.json          -- Schedules, preferences, defaults

Usage:
    from src.payment_processor import get_processor

    proc = get_processor()
    proc.record_payment(source=PaymentSource.ADSENSE, amount=142.50,
                        site_id="witchcraft", description="Feb 2026 AdSense")
    breakdown = proc.get_revenue_breakdown("2026-01-01", "2026-01-31")

CLI:
    python -m src.payment_processor record --source adsense --amount 142.50 --site witchcraft
    python -m src.payment_processor payments --source adsense --status received
    python -m src.payment_processor import --source adsense --file export.csv
    python -m src.payment_processor invoices --status paid
    python -m src.payment_processor create-invoice --client "Acme Corp" --email client@acme.com
    python -m src.payment_processor reconcile --source adsense --start 2026-01-01 --end 2026-01-31 --expected 500
    python -m src.payment_processor reconciliations --source adsense
    python -m src.payment_processor schedules
    python -m src.payment_processor create-schedule --source adsense --amount 150 --frequency monthly
    python -m src.payment_processor overdue
    python -m src.payment_processor expected --days 30
    python -m src.payment_processor breakdown --start 2026-01-01 --end 2026-01-31
    python -m src.payment_processor monthly --months 6
    python -m src.payment_processor fees --days 30
    python -m src.payment_processor tax --year 2025
    python -m src.payment_processor report --months 3
    python -m src.payment_processor stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("payment_processor")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "payments"
PAYMENTS_FILE = DATA_DIR / "payments.json"
INVOICES_FILE = DATA_DIR / "invoices.json"
RECONCILIATION_FILE = DATA_DIR / "reconciliation.json"
CONFIG_FILE = DATA_DIR / "config.json"

# Ensure directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]

SITE_DOMAIN_MAP: dict[str, str] = {
    "witchcraft": "witchcraftforbeginners.com",
    "smarthome": "smarthomewizards.com",
    "aiaction": "aiinactionhub.com",
    "aidiscovery": "aidiscoverydigest.com",
    "wealthai": "wealthfromai.com",
    "family": "family-flourish.com",
    "mythical": "mythicalarchives.com",
    "bulletjournals": "bulletjournals.net",
    "crystalwitchcraft": "crystalwitchcraft.com",
    "herbalwitchery": "herbalwitchery.com",
    "moonphasewitch": "moonphasewitch.com",
    "tarotbeginners": "tarotforbeginners.net",
    "spellsrituals": "spellsandrituals.com",
    "paganpathways": "paganpathways.net",
    "witchyhomedecor": "witchyhomedecor.com",
    "seasonalwitchcraft": "seasonalwitchcraft.com",
}

MAX_PAYMENTS = 50000
MAX_INVOICES = 10000
MAX_RECONCILIATIONS = 5000
MAX_SCHEDULES = 200

INVOICE_NUMBER_PREFIX = "INV"
DEFAULT_PAYMENT_TERMS = "Net 30"
DEFAULT_TAX_RATE = 0.0
DEFAULT_CURRENCY = "USD"

# Source-specific typical fee percentages (for estimation when not provided)
SOURCE_FEE_RATES: dict[str, float] = {
    "adsense": 0.0,           # Google pays net
    "amazon_affiliate": 0.0,  # Amazon pays net
    "other_affiliate": 0.0,
    "kdp_royalty": 0.0,       # Amazon withholds its share already
    "etsy_sales": 0.08,       # ~8% fees + processing
    "substack_paid": 0.10,    # 10% Substack fee
    "sponsorship": 0.0,
    "consulting": 0.029,      # ~2.9% Stripe/PayPal
    "digital_product": 0.05,  # platform-dependent
    "course": 0.05,
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PaymentSource(str, Enum):
    """Revenue source categories across the empire."""
    ADSENSE = "adsense"
    AMAZON_AFFILIATE = "amazon_affiliate"
    OTHER_AFFILIATE = "other_affiliate"
    KDP_ROYALTY = "kdp_royalty"
    ETSY_SALES = "etsy_sales"
    SUBSTACK_PAID = "substack_paid"
    SPONSORSHIP = "sponsorship"
    CONSULTING = "consulting"
    DIGITAL_PRODUCT = "digital_product"
    COURSE = "course"

    @classmethod
    def from_string(cls, value: str) -> PaymentSource:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(
            f"Unknown payment source: {value!r}. "
            f"Valid: {', '.join(m.value for m in cls)}"
        )


class PaymentStatus(str, Enum):
    """Lifecycle status of a payment."""
    PENDING = "pending"
    RECEIVED = "received"
    CLEARED = "cleared"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"

    @classmethod
    def from_string(cls, value: str) -> PaymentStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown payment status: {value!r}")


class InvoiceStatus(str, Enum):
    """Lifecycle status of an invoice."""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    VOIDED = "voided"

    @classmethod
    def from_string(cls, value: str) -> InvoiceStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown invoice status: {value!r}")


class ReconciliationStatus(str, Enum):
    """Outcome of a reconciliation check."""
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    DISCREPANCY = "discrepancy"
    PENDING = "pending"

    @classmethod
    def from_string(cls, value: str) -> ReconciliationStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown reconciliation status: {value!r}")


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"

    @classmethod
    def from_string(cls, value: str) -> Currency:
        normalized = value.strip().upper()
        for member in cls:
            if member.value == normalized or member.name == normalized:
                return member
        raise ValueError(f"Unknown currency: {value!r}")


class PaymentFrequency(str, Enum):
    """How often a payment is expected."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ONE_TIME = "one_time"

    @classmethod
    def from_string(cls, value: str) -> PaymentFrequency:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown payment frequency: {value!r}")

    @property
    def days(self) -> int:
        """Approximate number of days between payments."""
        return {
            PaymentFrequency.DAILY: 1,
            PaymentFrequency.WEEKLY: 7,
            PaymentFrequency.BIWEEKLY: 14,
            PaymentFrequency.MONTHLY: 30,
            PaymentFrequency.QUARTERLY: 90,
            PaymentFrequency.ANNUAL: 365,
            PaymentFrequency.ONE_TIME: 0,
        }[self]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _parse_date(d: str) -> date:
    """Parse an ISO date string (YYYY-MM-DD)."""
    return date.fromisoformat(d)


def _parse_datetime(d: str) -> datetime:
    """Parse an ISO datetime string."""
    return datetime.fromisoformat(d)


def _round_amount(amount: float) -> float:
    return round(float(amount), 2)


def _gen_id(prefix: str = "pay") -> str:
    """Generate a unique ID with prefix."""
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}"


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from path, returning default when missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write data as pretty-printed JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
    os.replace(str(tmp), str(path))


def _run_sync(coro):
    """Run an async coroutine from synchronous code, handling nested loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _month_bounds(ref: Optional[date] = None) -> tuple[str, str]:
    """Return (first_day, last_day) ISO strings for the month of ref."""
    if ref is None:
        ref = _now_utc().date()
    first = ref.replace(day=1)
    if ref.month == 12:
        last = ref.replace(year=ref.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        last = ref.replace(month=ref.month + 1, day=1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def _months_ago(months: int) -> str:
    """Return ISO date string for N months ago (approximate)."""
    d = _now_utc().date() - timedelta(days=months * 30)
    return d.isoformat()


def _days_between(d1: str, d2: str) -> int:
    """Return number of days between two ISO date strings."""
    return abs((_parse_date(d2) - _parse_date(d1)).days)


def _format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount with currency symbol."""
    symbols = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CAD": "C$", "AUD": "A$"}
    sym = symbols.get(currency, currency + " ")
    return f"{sym}{amount:,.2f}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Payment:
    """A single payment received or expected."""
    payment_id: str = ""
    source: str = ""
    amount: float = 0.0
    currency: str = DEFAULT_CURRENCY
    status: str = PaymentStatus.PENDING.value
    reference_id: str = ""
    site_id: str = ""
    description: str = ""
    received_date: str = ""
    cleared_date: str = ""
    payment_method: str = ""
    fees: float = 0.0
    net_amount: float = 0.0
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.payment_id:
            self.payment_id = _gen_id("pay")
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.received_date:
            self.received_date = _today_iso()
        if self.net_amount == 0.0 and self.amount > 0:
            self.net_amount = _round_amount(self.amount - self.fees)

    @classmethod
    def from_dict(cls, data: dict) -> Payment:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        p = cls(**filtered)
        return p

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InvoiceItem:
    """Single line item on an invoice."""
    description: str = ""
    quantity: float = 1.0
    unit_price: float = 0.0
    total: float = 0.0

    def __post_init__(self):
        if self.total == 0.0:
            self.total = _round_amount(self.quantity * self.unit_price)


@dataclass
class Invoice:
    """An invoice for client billing (sponsorships, consulting, etc)."""
    invoice_id: str = ""
    client_name: str = ""
    client_email: str = ""
    items: list = field(default_factory=list)
    subtotal: float = 0.0
    tax_rate: float = DEFAULT_TAX_RATE
    tax_amount: float = 0.0
    total: float = 0.0
    currency: str = DEFAULT_CURRENCY
    status: str = InvoiceStatus.DRAFT.value
    issued_date: str = ""
    due_date: str = ""
    paid_date: str = ""
    notes: str = ""
    payment_terms: str = DEFAULT_PAYMENT_TERMS
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.invoice_id:
            self.invoice_id = _gen_id("inv")
        if not self.issued_date:
            self.issued_date = _today_iso()
        if not self.due_date:
            due = _now_utc().date() + timedelta(days=30)
            self.due_date = due.isoformat()
        self._recalculate()

    def _recalculate(self):
        """Recalculate subtotal, tax, and total from items."""
        if self.items:
            self.subtotal = _round_amount(
                sum(
                    (item.get("total", 0) if isinstance(item, dict)
                     else item.total if hasattr(item, "total") else 0)
                    for item in self.items
                )
            )
        self.tax_amount = _round_amount(self.subtotal * self.tax_rate)
        self.total = _round_amount(self.subtotal + self.tax_amount)

    @classmethod
    def from_dict(cls, data: dict) -> Invoice:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        inv = cls(**filtered)
        return inv

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReconciliationRecord:
    """Result of reconciling expected vs actual revenue for a source/period."""
    record_id: str = ""
    period_start: str = ""
    period_end: str = ""
    source: str = ""
    expected_amount: float = 0.0
    actual_amount: float = 0.0
    discrepancy: float = 0.0
    status: str = ReconciliationStatus.PENDING.value
    payments_matched: list = field(default_factory=list)
    payments_unmatched: list = field(default_factory=list)
    notes: str = ""
    reconciled_at: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = _gen_id("rec")
        if not self.reconciled_at:
            self.reconciled_at = _now_iso()
        self.discrepancy = _round_amount(self.actual_amount - self.expected_amount)

    @classmethod
    def from_dict(cls, data: dict) -> ReconciliationRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaymentSchedule:
    """Expected recurring payment configuration."""
    schedule_id: str = ""
    source: str = ""
    expected_amount: float = 0.0
    frequency: str = PaymentFrequency.MONTHLY.value
    next_expected_date: str = ""
    last_received_date: str = ""
    account_details: dict = field(default_factory=dict)
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.schedule_id:
            self.schedule_id = _gen_id("sch")
        if not self.next_expected_date:
            freq = PaymentFrequency.from_string(self.frequency)
            days_ahead = freq.days if freq.days > 0 else 30
            next_d = _now_utc().date() + timedelta(days=days_ahead)
            self.next_expected_date = next_d.isoformat()

    @classmethod
    def from_dict(cls, data: dict) -> PaymentSchedule:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RevenueBreakdown:
    """Aggregated revenue data for a period."""
    period: str = ""
    total_gross: float = 0.0
    total_fees: float = 0.0
    total_net: float = 0.0
    by_source: dict = field(default_factory=dict)
    by_site: dict = field(default_factory=dict)
    by_currency: dict = field(default_factory=dict)
    payment_count: int = 0
    avg_payment: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Invoice HTML Template
# ---------------------------------------------------------------------------

INVOICE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Invoice {invoice_number}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 40px; color: #333; background: #f5f5f5; }}
  .invoice {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 40px; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; }}
  .company {{ font-size: 24px; font-weight: 700; color: #2c3e50; }}
  .company-details {{ font-size: 13px; color: #666; margin-top: 5px; }}
  .invoice-meta {{ text-align: right; }}
  .invoice-meta h2 {{ margin: 0; color: #2c3e50; font-size: 28px; }}
  .invoice-meta p {{ margin: 4px 0; font-size: 13px; color: #666; }}
  .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
  .status-draft {{ background: #f0f0f0; color: #666; }}
  .status-sent {{ background: #e3f2fd; color: #1565c0; }}
  .status-paid {{ background: #e8f5e9; color: #2e7d32; }}
  .status-overdue {{ background: #fce4ec; color: #c62828; }}
  .status-cancelled {{ background: #fafafa; color: #999; }}
  .status-voided {{ background: #fafafa; color: #999; text-decoration: line-through; }}
  .client {{ margin-bottom: 30px; }}
  .client h3 {{ margin: 0 0 8px 0; color: #2c3e50; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
  .client p {{ margin: 2px 0; font-size: 14px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
  th {{ background: #2c3e50; color: #fff; padding: 12px 15px; text-align: left; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
  th:last-child {{ text-align: right; }}
  td {{ padding: 12px 15px; border-bottom: 1px solid #eee; font-size: 14px; }}
  td:last-child {{ text-align: right; }}
  .totals {{ display: flex; justify-content: flex-end; }}
  .totals table {{ width: 300px; }}
  .totals td {{ border: none; padding: 6px 15px; }}
  .totals .total-row td {{ font-weight: 700; font-size: 18px; border-top: 2px solid #2c3e50; padding-top: 12px; }}
  .notes {{ margin-top: 30px; padding: 20px; background: #f9f9f9; border-radius: 4px; font-size: 13px; color: #666; }}
  .notes h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #2c3e50; }}
  .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #999; border-top: 1px solid #eee; padding-top: 20px; }}
</style>
</head>
<body>
<div class="invoice">
  <div class="header">
    <div>
      <div class="company">OpenClaw Empire</div>
      <div class="company-details">
        Nick Creighton<br>
        Digital Publishing &amp; Automation<br>
        empire@openclaw.dev
      </div>
    </div>
    <div class="invoice-meta">
      <h2>INVOICE</h2>
      <p><strong>{invoice_number}</strong></p>
      <p>Issued: {issued_date}</p>
      <p>Due: {due_date}</p>
      <p><span class="status status-{status_class}">{status}</span></p>
    </div>
  </div>
  <div class="client">
    <h3>Bill To</h3>
    <p><strong>{client_name}</strong></p>
    <p>{client_email}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Description</th>
        <th>Qty</th>
        <th>Unit Price</th>
        <th>Total</th>
      </tr>
    </thead>
    <tbody>
      {items_html}
    </tbody>
  </table>
  <div class="totals">
    <table>
      <tr><td>Subtotal</td><td>{subtotal}</td></tr>
      {tax_row}
      <tr class="total-row"><td>Total ({currency})</td><td>{total}</td></tr>
    </table>
  </div>
  {notes_html}
  <div class="footer">
    <p>Payment Terms: {payment_terms}</p>
    <p>Thank you for your business.</p>
  </div>
</div>
</body>
</html>"""


# ===========================================================================
# PaymentProcessor — Singleton
# ===========================================================================

class PaymentProcessor:
    """
    Central payment processing engine for the OpenClaw Empire.

    Manages payments, invoices, reconciliation records, and payment schedules.
    All data is persisted to JSON files under data/payments/.
    """

    def __init__(self) -> None:
        self._payments: dict[str, dict] = {}
        self._invoices: dict[str, dict] = {}
        self._reconciliations: dict[str, dict] = {}
        self._config: dict[str, Any] = {}
        self._load_all()
        logger.info(
            "PaymentProcessor initialized: %d payments, %d invoices, "
            "%d reconciliations, %d schedules",
            len(self._payments), len(self._invoices),
            len(self._reconciliations), len(self.list_schedules()),
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all data files from disk."""
        raw_payments = _load_json(PAYMENTS_FILE, default={})
        if isinstance(raw_payments, list):
            self._payments = {p["payment_id"]: p for p in raw_payments if "payment_id" in p}
        else:
            self._payments = raw_payments

        raw_invoices = _load_json(INVOICES_FILE, default={})
        if isinstance(raw_invoices, list):
            self._invoices = {i["invoice_id"]: i for i in raw_invoices if "invoice_id" in i}
        else:
            self._invoices = raw_invoices

        raw_recon = _load_json(RECONCILIATION_FILE, default={})
        if isinstance(raw_recon, list):
            self._reconciliations = {r["record_id"]: r for r in raw_recon if "record_id" in r}
        else:
            self._reconciliations = raw_recon

        self._config = _load_json(CONFIG_FILE, default={
            "schedules": {},
            "invoice_counter": 0,
            "default_currency": DEFAULT_CURRENCY,
            "default_tax_rate": DEFAULT_TAX_RATE,
            "default_payment_terms": DEFAULT_PAYMENT_TERMS,
        })
        if "schedules" not in self._config:
            self._config["schedules"] = {}
        if "invoice_counter" not in self._config:
            self._config["invoice_counter"] = 0

    def _save_payments(self) -> None:
        _save_json(PAYMENTS_FILE, self._payments)

    def _save_invoices(self) -> None:
        _save_json(INVOICES_FILE, self._invoices)

    def _save_reconciliations(self) -> None:
        _save_json(RECONCILIATION_FILE, self._reconciliations)

    def _save_config(self) -> None:
        _save_json(CONFIG_FILE, self._config)

    # ===================================================================
    # PAYMENTS
    # ===================================================================

    def record_payment(
        self,
        source: str | PaymentSource,
        amount: float,
        *,
        currency: str = DEFAULT_CURRENCY,
        status: str = PaymentStatus.RECEIVED.value,
        reference_id: str = "",
        site_id: str = "",
        description: str = "",
        received_date: str = "",
        cleared_date: str = "",
        payment_method: str = "",
        fees: float = 0.0,
        net_amount: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> Payment:
        """Record a new payment."""
        if isinstance(source, PaymentSource):
            source_val = source.value
        else:
            source_val = PaymentSource.from_string(source).value

        if site_id and site_id not in ALL_SITE_IDS:
            logger.warning("Unknown site_id '%s' — recording anyway", site_id)

        amount = _round_amount(amount)
        fees = _round_amount(fees)

        # Auto-estimate fees if not provided
        if fees == 0.0 and source_val in SOURCE_FEE_RATES:
            rate = SOURCE_FEE_RATES[source_val]
            if rate > 0:
                fees = _round_amount(amount * rate)

        if net_amount == 0.0:
            net_amount = _round_amount(amount - fees)

        payment = Payment(
            source=source_val,
            amount=amount,
            currency=currency,
            status=status,
            reference_id=reference_id,
            site_id=site_id,
            description=description,
            received_date=received_date or _today_iso(),
            cleared_date=cleared_date,
            payment_method=payment_method,
            fees=fees,
            net_amount=net_amount,
            metadata=metadata or {},
        )

        if len(self._payments) >= MAX_PAYMENTS:
            logger.warning("Payment store at capacity (%d). Pruning oldest.", MAX_PAYMENTS)
            self._prune_payments()

        self._payments[payment.payment_id] = payment.to_dict()
        self._save_payments()

        logger.info(
            "Recorded payment %s: %s %s from %s (net: %s)",
            payment.payment_id, _format_currency(amount, currency),
            source_val, site_id or "general", _format_currency(net_amount, currency),
        )
        return payment

    def update_payment(self, payment_id: str, **kwargs) -> Payment:
        """Update fields on an existing payment."""
        if payment_id not in self._payments:
            raise KeyError(f"Payment not found: {payment_id}")

        data = self._payments[payment_id]

        if "source" in kwargs:
            kwargs["source"] = PaymentSource.from_string(kwargs["source"]).value
        if "status" in kwargs:
            kwargs["status"] = PaymentStatus.from_string(kwargs["status"]).value
        if "amount" in kwargs:
            kwargs["amount"] = _round_amount(kwargs["amount"])
        if "fees" in kwargs:
            kwargs["fees"] = _round_amount(kwargs["fees"])
        if "net_amount" in kwargs:
            kwargs["net_amount"] = _round_amount(kwargs["net_amount"])

        for key, value in kwargs.items():
            if key in data:
                data[key] = value

        # Recalculate net if amount or fees changed
        if "amount" in kwargs or "fees" in kwargs:
            data["net_amount"] = _round_amount(data["amount"] - data["fees"])

        self._payments[payment_id] = data
        self._save_payments()

        logger.info("Updated payment %s: %s", payment_id, list(kwargs.keys()))
        return Payment.from_dict(data)

    def get_payment(self, payment_id: str) -> Payment:
        """Retrieve a single payment by ID."""
        if payment_id not in self._payments:
            raise KeyError(f"Payment not found: {payment_id}")
        return Payment.from_dict(self._payments[payment_id])

    def list_payments(
        self,
        source: Optional[str] = None,
        status: Optional[str] = None,
        site_id: Optional[str] = None,
        date_range: Optional[tuple[str, str]] = None,
        currency: Optional[str] = None,
        limit: int = 0,
    ) -> list[Payment]:
        """List payments with optional filters."""
        results = []

        source_val = PaymentSource.from_string(source).value if source else None
        status_val = PaymentStatus.from_string(status).value if status else None

        for data in self._payments.values():
            if source_val and data.get("source") != source_val:
                continue
            if status_val and data.get("status") != status_val:
                continue
            if site_id and data.get("site_id") != site_id:
                continue
            if currency and data.get("currency") != currency.upper():
                continue
            if date_range:
                rd = data.get("received_date", "")
                if rd and (rd < date_range[0] or rd > date_range[1]):
                    continue
            results.append(Payment.from_dict(data))

        # Sort by received_date descending (most recent first)
        results.sort(key=lambda p: p.received_date, reverse=True)

        if limit > 0:
            results = results[:limit]

        return results

    def delete_payment(self, payment_id: str) -> bool:
        """Delete a payment record. Returns True if found and deleted."""
        if payment_id not in self._payments:
            return False
        del self._payments[payment_id]
        self._save_payments()
        logger.info("Deleted payment %s", payment_id)
        return True

    async def import_payments(
        self, source: str, data: list[dict],
    ) -> list[Payment]:
        """
        Bulk import payments from an external platform export.

        Each dict in data should contain at minimum: amount.
        Optional: reference_id, received_date, description, site_id, fees.
        """
        source_val = PaymentSource.from_string(source).value
        imported = []

        for entry in data:
            amount = entry.get("amount", 0)
            if not amount:
                logger.warning("Skipping entry with zero amount: %s", entry)
                continue

            payment = self.record_payment(
                source=source_val,
                amount=float(amount),
                reference_id=str(entry.get("reference_id", entry.get("id", ""))),
                site_id=entry.get("site_id", ""),
                description=entry.get("description", f"Imported from {source_val}"),
                received_date=entry.get("received_date", entry.get("date", "")),
                fees=float(entry.get("fees", 0)),
                payment_method=entry.get("payment_method", ""),
                metadata={"imported": True, "import_source": source_val},
            )
            imported.append(payment)

        logger.info(
            "Imported %d payments from %s (total: %s)",
            len(imported), source_val,
            _format_currency(sum(p.amount for p in imported)),
        )
        return imported

    def _prune_payments(self, keep: int = 40000) -> int:
        """Remove oldest payments to stay under capacity."""
        all_payments = sorted(
            self._payments.items(),
            key=lambda kv: kv[1].get("created_at", ""),
        )
        to_remove = len(all_payments) - keep
        if to_remove <= 0:
            return 0
        for pid, _ in all_payments[:to_remove]:
            del self._payments[pid]
        self._save_payments()
        logger.info("Pruned %d old payments", to_remove)
        return to_remove

    # ===================================================================
    # INVOICES
    # ===================================================================

    def generate_invoice_number(self) -> str:
        """Generate next sequential invoice number like INV-2026-0001."""
        self._config["invoice_counter"] = self._config.get("invoice_counter", 0) + 1
        counter = self._config["invoice_counter"]
        self._save_config()
        year = _now_utc().year
        return f"{INVOICE_NUMBER_PREFIX}-{year}-{counter:04d}"

    def create_invoice(
        self,
        client_name: str,
        client_email: str,
        items: list[dict],
        *,
        currency: str = DEFAULT_CURRENCY,
        tax_rate: float = DEFAULT_TAX_RATE,
        notes: str = "",
        payment_terms: str = "",
        due_days: int = 30,
        metadata: Optional[dict] = None,
    ) -> Invoice:
        """Create a new invoice."""
        if not client_name:
            raise ValueError("client_name is required")
        if not items:
            raise ValueError("At least one item is required")

        # Validate and normalize items
        normalized_items = []
        for item in items:
            desc = item.get("description", "Service")
            qty = float(item.get("quantity", 1))
            price = float(item.get("unit_price", 0))
            total = _round_amount(qty * price)
            normalized_items.append({
                "description": desc,
                "quantity": qty,
                "unit_price": _round_amount(price),
                "total": total,
            })

        due_date = (_now_utc().date() + timedelta(days=due_days)).isoformat()

        invoice = Invoice(
            client_name=client_name,
            client_email=client_email,
            items=normalized_items,
            currency=currency,
            tax_rate=tax_rate,
            notes=notes,
            payment_terms=payment_terms or self._config.get(
                "default_payment_terms", DEFAULT_PAYMENT_TERMS
            ),
            due_date=due_date,
            metadata=metadata or {},
        )
        invoice._recalculate()

        # Assign human-readable invoice number
        inv_number = self.generate_invoice_number()
        invoice.metadata["invoice_number"] = inv_number

        if len(self._invoices) >= MAX_INVOICES:
            logger.warning("Invoice store at capacity.")

        self._invoices[invoice.invoice_id] = invoice.to_dict()
        self._save_invoices()

        logger.info(
            "Created invoice %s (%s) for %s: %s",
            invoice.invoice_id, inv_number, client_name,
            _format_currency(invoice.total, currency),
        )
        return invoice

    def update_invoice(self, invoice_id: str, **kwargs) -> Invoice:
        """Update fields on an existing invoice."""
        if invoice_id not in self._invoices:
            raise KeyError(f"Invoice not found: {invoice_id}")

        data = self._invoices[invoice_id]

        if "status" in kwargs:
            kwargs["status"] = InvoiceStatus.from_string(kwargs["status"]).value

        for key, value in kwargs.items():
            if key in data:
                data[key] = value

        # Recalculate totals if items or tax changed
        if "items" in kwargs or "tax_rate" in kwargs:
            inv = Invoice.from_dict(data)
            inv._recalculate()
            data = inv.to_dict()

        self._invoices[invoice_id] = data
        self._save_invoices()
        logger.info("Updated invoice %s: %s", invoice_id, list(kwargs.keys()))
        return Invoice.from_dict(data)

    def send_invoice(self, invoice_id: str) -> Invoice:
        """Mark an invoice as sent."""
        return self.update_invoice(invoice_id, status=InvoiceStatus.SENT.value)

    def mark_paid(self, invoice_id: str, paid_date: str = "") -> Invoice:
        """Mark an invoice as paid."""
        return self.update_invoice(
            invoice_id,
            status=InvoiceStatus.PAID.value,
            paid_date=paid_date or _today_iso(),
        )

    def void_invoice(self, invoice_id: str) -> Invoice:
        """Void an invoice (cannot be undone)."""
        return self.update_invoice(invoice_id, status=InvoiceStatus.VOIDED.value)

    def get_invoice(self, invoice_id: str) -> Invoice:
        """Retrieve a single invoice by ID."""
        if invoice_id not in self._invoices:
            raise KeyError(f"Invoice not found: {invoice_id}")
        return Invoice.from_dict(self._invoices[invoice_id])

    def list_invoices(
        self,
        status: Optional[str] = None,
        client_name: Optional[str] = None,
        date_range: Optional[tuple[str, str]] = None,
        limit: int = 0,
    ) -> list[Invoice]:
        """List invoices with optional filters."""
        results = []
        status_val = InvoiceStatus.from_string(status).value if status else None

        for data in self._invoices.values():
            if status_val and data.get("status") != status_val:
                continue
            if client_name and client_name.lower() not in data.get("client_name", "").lower():
                continue
            if date_range:
                issued = data.get("issued_date", "")
                if issued and (issued < date_range[0] or issued > date_range[1]):
                    continue
            results.append(Invoice.from_dict(data))

        results.sort(key=lambda i: i.issued_date, reverse=True)
        if limit > 0:
            results = results[:limit]
        return results

    async def generate_invoice_html(self, invoice_id: str) -> str:
        """Generate a professional HTML invoice from template."""
        inv = self.get_invoice(invoice_id)
        inv_number = inv.metadata.get("invoice_number", inv.invoice_id)

        items_rows = []
        for item in inv.items:
            if isinstance(item, dict):
                desc = item.get("description", "")
                qty = item.get("quantity", 1)
                price = item.get("unit_price", 0)
                total = item.get("total", 0)
            else:
                desc = getattr(item, "description", "")
                qty = getattr(item, "quantity", 1)
                price = getattr(item, "unit_price", 0)
                total = getattr(item, "total", 0)

            items_rows.append(
                f"      <tr>"
                f"<td>{desc}</td>"
                f"<td>{qty}</td>"
                f"<td>{_format_currency(price, inv.currency)}</td>"
                f"<td>{_format_currency(total, inv.currency)}</td>"
                f"</tr>"
            )

        items_html = "\n".join(items_rows)

        tax_row = ""
        if inv.tax_rate > 0:
            tax_row = (
                f"<tr><td>Tax ({inv.tax_rate * 100:.1f}%)</td>"
                f"<td>{_format_currency(inv.tax_amount, inv.currency)}</td></tr>"
            )

        notes_html = ""
        if inv.notes:
            notes_html = (
                f'<div class="notes"><h3>Notes</h3><p>{inv.notes}</p></div>'
            )

        status_class = inv.status.lower()

        html = INVOICE_HTML_TEMPLATE.format(
            invoice_number=inv_number,
            issued_date=inv.issued_date,
            due_date=inv.due_date,
            status=inv.status.upper(),
            status_class=status_class,
            client_name=inv.client_name,
            client_email=inv.client_email,
            items_html=items_html,
            subtotal=_format_currency(inv.subtotal, inv.currency),
            tax_row=tax_row,
            total=_format_currency(inv.total, inv.currency),
            currency=inv.currency,
            payment_terms=inv.payment_terms,
            notes_html=notes_html,
        )
        return html

    # ===================================================================
    # RECONCILIATION
    # ===================================================================

    async def reconcile(
        self,
        source: str,
        period_start: str,
        period_end: str,
        expected_amount: float,
    ) -> ReconciliationRecord:
        """
        Reconcile expected vs actual revenue for a source in a given period.

        Finds all matching payments in the date range, calculates actual total,
        and flags discrepancies.
        """
        source_val = PaymentSource.from_string(source).value
        expected_amount = _round_amount(expected_amount)

        # Find all payments matching source and date range
        matching = self.list_payments(
            source=source_val,
            date_range=(period_start, period_end),
        )

        # Split by status — received/cleared are matched, others unmatched
        matched_ids = []
        unmatched_ids = []
        actual_total = 0.0

        for p in matching:
            if p.status in (PaymentStatus.RECEIVED.value, PaymentStatus.CLEARED.value):
                matched_ids.append(p.payment_id)
                actual_total += p.net_amount
            else:
                unmatched_ids.append(p.payment_id)

        actual_total = _round_amount(actual_total)
        discrepancy = _round_amount(actual_total - expected_amount)

        # Determine reconciliation status
        abs_disc = abs(discrepancy)
        if abs_disc < 0.01:
            recon_status = ReconciliationStatus.MATCHED.value
        elif not matched_ids and not unmatched_ids:
            recon_status = ReconciliationStatus.UNMATCHED.value
        elif abs_disc > 0:
            recon_status = ReconciliationStatus.DISCREPANCY.value
        else:
            recon_status = ReconciliationStatus.MATCHED.value

        notes = ""
        if discrepancy > 0:
            notes = f"Received {_format_currency(abs_disc)} MORE than expected."
        elif discrepancy < 0:
            notes = f"Received {_format_currency(abs_disc)} LESS than expected."
        else:
            notes = "Perfectly matched."

        record = ReconciliationRecord(
            period_start=period_start,
            period_end=period_end,
            source=source_val,
            expected_amount=expected_amount,
            actual_amount=actual_total,
            discrepancy=discrepancy,
            status=recon_status,
            payments_matched=matched_ids,
            payments_unmatched=unmatched_ids,
            notes=notes,
        )

        if len(self._reconciliations) >= MAX_RECONCILIATIONS:
            self._prune_reconciliations()

        self._reconciliations[record.record_id] = record.to_dict()
        self._save_reconciliations()

        logger.info(
            "Reconciliation %s: %s %s..%s expected=%s actual=%s disc=%s [%s]",
            record.record_id, source_val, period_start, period_end,
            _format_currency(expected_amount), _format_currency(actual_total),
            _format_currency(discrepancy), recon_status,
        )
        return record

    def get_reconciliation(self, record_id: str) -> ReconciliationRecord:
        """Retrieve a single reconciliation record."""
        if record_id not in self._reconciliations:
            raise KeyError(f"Reconciliation record not found: {record_id}")
        return ReconciliationRecord.from_dict(self._reconciliations[record_id])

    def list_reconciliations(
        self,
        source: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 0,
    ) -> list[ReconciliationRecord]:
        """List reconciliation records with optional filters."""
        results = []
        source_val = PaymentSource.from_string(source).value if source else None
        status_val = ReconciliationStatus.from_string(status).value if status else None

        for data in self._reconciliations.values():
            if source_val and data.get("source") != source_val:
                continue
            if status_val and data.get("status") != status_val:
                continue
            results.append(ReconciliationRecord.from_dict(data))

        results.sort(key=lambda r: r.reconciled_at, reverse=True)
        if limit > 0:
            results = results[:limit]
        return results

    def get_discrepancies(self, days: int = 30) -> list[ReconciliationRecord]:
        """Get all reconciliation records with discrepancies within N days."""
        cutoff = (_now_utc().date() - timedelta(days=days)).isoformat()
        results = []
        for data in self._reconciliations.values():
            if data.get("status") == ReconciliationStatus.DISCREPANCY.value:
                if data.get("period_end", "") >= cutoff:
                    results.append(ReconciliationRecord.from_dict(data))
        results.sort(key=lambda r: abs(r.discrepancy), reverse=True)
        return results

    def _prune_reconciliations(self, keep: int = 4000) -> int:
        """Remove oldest reconciliation records."""
        all_recs = sorted(
            self._reconciliations.items(),
            key=lambda kv: kv[1].get("reconciled_at", ""),
        )
        to_remove = len(all_recs) - keep
        if to_remove <= 0:
            return 0
        for rid, _ in all_recs[:to_remove]:
            del self._reconciliations[rid]
        self._save_reconciliations()
        return to_remove

    # ===================================================================
    # SCHEDULES
    # ===================================================================

    def create_schedule(
        self,
        source: str,
        expected_amount: float,
        frequency: str = "monthly",
        *,
        next_expected_date: str = "",
        account_details: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> PaymentSchedule:
        """Create a new payment schedule (expected recurring payment)."""
        source_val = PaymentSource.from_string(source).value
        freq_val = PaymentFrequency.from_string(frequency).value

        schedule = PaymentSchedule(
            source=source_val,
            expected_amount=_round_amount(expected_amount),
            frequency=freq_val,
            next_expected_date=next_expected_date,
            account_details=account_details or {},
            metadata=metadata or {},
        )

        schedules = self._config.get("schedules", {})
        if len(schedules) >= MAX_SCHEDULES:
            raise ValueError(f"Maximum schedules ({MAX_SCHEDULES}) reached.")

        schedules[schedule.schedule_id] = schedule.to_dict()
        self._config["schedules"] = schedules
        self._save_config()

        logger.info(
            "Created schedule %s: %s %s every %s, next: %s",
            schedule.schedule_id, source_val,
            _format_currency(expected_amount), freq_val,
            schedule.next_expected_date,
        )
        return schedule

    def update_schedule(self, schedule_id: str, **kwargs) -> PaymentSchedule:
        """Update fields on a payment schedule."""
        schedules = self._config.get("schedules", {})
        if schedule_id not in schedules:
            raise KeyError(f"Schedule not found: {schedule_id}")

        data = schedules[schedule_id]

        if "source" in kwargs:
            kwargs["source"] = PaymentSource.from_string(kwargs["source"]).value
        if "frequency" in kwargs:
            kwargs["frequency"] = PaymentFrequency.from_string(kwargs["frequency"]).value
        if "expected_amount" in kwargs:
            kwargs["expected_amount"] = _round_amount(kwargs["expected_amount"])

        for key, value in kwargs.items():
            if key in data:
                data[key] = value

        schedules[schedule_id] = data
        self._config["schedules"] = schedules
        self._save_config()
        logger.info("Updated schedule %s: %s", schedule_id, list(kwargs.keys()))
        return PaymentSchedule.from_dict(data)

    def get_schedule(self, schedule_id: str) -> PaymentSchedule:
        """Retrieve a single payment schedule."""
        schedules = self._config.get("schedules", {})
        if schedule_id not in schedules:
            raise KeyError(f"Schedule not found: {schedule_id}")
        return PaymentSchedule.from_dict(schedules[schedule_id])

    def list_schedules(self, active_only: bool = False) -> list[PaymentSchedule]:
        """List all payment schedules."""
        schedules = self._config.get("schedules", {})
        results = []
        for data in schedules.values():
            sched = PaymentSchedule.from_dict(data)
            if active_only and not sched.active:
                continue
            results.append(sched)
        results.sort(key=lambda s: s.next_expected_date)
        return results

    def check_overdue_payments(self) -> list[PaymentSchedule]:
        """
        Find all active schedules whose next_expected_date has passed
        without a matching payment.
        """
        today = _today_iso()
        overdue = []

        for sched in self.list_schedules(active_only=True):
            if sched.next_expected_date and sched.next_expected_date < today:
                # Check if a payment was received on or after expected date
                freq = PaymentFrequency.from_string(sched.frequency)
                lookback = max(freq.days, 7) if freq.days > 0 else 30
                lookback_start = (
                    _parse_date(sched.next_expected_date) - timedelta(days=lookback)
                ).isoformat()

                recent = self.list_payments(
                    source=sched.source,
                    date_range=(lookback_start, today),
                )
                received = [
                    p for p in recent
                    if p.status in (
                        PaymentStatus.RECEIVED.value,
                        PaymentStatus.CLEARED.value,
                    )
                ]

                if not received:
                    overdue.append(sched)

        return overdue

    def get_expected_income(self, days: int = 30) -> list[dict]:
        """
        Project expected income for the next N days based on active schedules.
        Returns list of dicts with source, amount, expected_date, frequency.
        """
        today = _now_utc().date()
        end = today + timedelta(days=days)
        projections = []

        for sched in self.list_schedules(active_only=True):
            if not sched.next_expected_date:
                continue

            freq = PaymentFrequency.from_string(sched.frequency)
            if freq == PaymentFrequency.ONE_TIME:
                next_d = _parse_date(sched.next_expected_date)
                if today <= next_d <= end:
                    projections.append({
                        "schedule_id": sched.schedule_id,
                        "source": sched.source,
                        "amount": sched.expected_amount,
                        "expected_date": sched.next_expected_date,
                        "frequency": sched.frequency,
                    })
                continue

            # Recurring: project all occurrences in window
            current = _parse_date(sched.next_expected_date)
            step = timedelta(days=freq.days) if freq.days > 0 else timedelta(days=30)

            # Walk backwards if current is past
            while current < today:
                current += step

            # Walk forward through window
            while current <= end:
                projections.append({
                    "schedule_id": sched.schedule_id,
                    "source": sched.source,
                    "amount": sched.expected_amount,
                    "expected_date": current.isoformat(),
                    "frequency": sched.frequency,
                })
                current += step

        projections.sort(key=lambda p: p["expected_date"])
        return projections

    # ===================================================================
    # ANALYTICS
    # ===================================================================

    def get_revenue_breakdown(
        self, period_start: str, period_end: str,
    ) -> RevenueBreakdown:
        """
        Get detailed revenue breakdown for a date range.
        Aggregates by source, site, and currency.
        """
        payments = self.list_payments(date_range=(period_start, period_end))

        # Only count received/cleared
        active = [
            p for p in payments
            if p.status in (
                PaymentStatus.RECEIVED.value,
                PaymentStatus.CLEARED.value,
            )
        ]

        total_gross = 0.0
        total_fees = 0.0
        total_net = 0.0
        by_source: dict[str, float] = {}
        by_site: dict[str, float] = {}
        by_currency: dict[str, float] = {}

        for p in active:
            total_gross += p.amount
            total_fees += p.fees
            total_net += p.net_amount

            by_source[p.source] = by_source.get(p.source, 0.0) + p.net_amount
            if p.site_id:
                by_site[p.site_id] = by_site.get(p.site_id, 0.0) + p.net_amount
            by_currency[p.currency] = by_currency.get(p.currency, 0.0) + p.net_amount

        count = len(active)
        avg = _round_amount(total_net / count) if count > 0 else 0.0

        breakdown = RevenueBreakdown(
            period=f"{period_start} to {period_end}",
            total_gross=_round_amount(total_gross),
            total_fees=_round_amount(total_fees),
            total_net=_round_amount(total_net),
            by_source={k: _round_amount(v) for k, v in sorted(
                by_source.items(), key=lambda kv: kv[1], reverse=True
            )},
            by_site={k: _round_amount(v) for k, v in sorted(
                by_site.items(), key=lambda kv: kv[1], reverse=True
            )},
            by_currency={k: _round_amount(v) for k, v in by_currency.items()},
            payment_count=count,
            avg_payment=avg,
        )
        return breakdown

    def get_monthly_summary(self, months: int = 12) -> list[dict]:
        """
        Get month-by-month revenue summary for the last N months.
        Returns list of dicts with month, gross, fees, net, count.
        """
        summaries = []
        today = _now_utc().date()

        for i in range(months):
            # Calculate month offset
            target_month = today.month - i
            target_year = today.year
            while target_month <= 0:
                target_month += 12
                target_year -= 1

            ref = date(target_year, target_month, 1)
            start, end = _month_bounds(ref)

            breakdown = self.get_revenue_breakdown(start, end)
            summaries.append({
                "month": ref.strftime("%Y-%m"),
                "gross": breakdown.total_gross,
                "fees": breakdown.total_fees,
                "net": breakdown.total_net,
                "count": breakdown.payment_count,
                "by_source": breakdown.by_source,
            })

        # Chronological order (oldest first)
        summaries.reverse()
        return summaries

    def get_source_trends(
        self, source: str, months: int = 6,
    ) -> list[dict]:
        """
        Get month-by-month revenue trend for a specific source.
        Returns list of dicts with month, amount, count, avg.
        """
        source_val = PaymentSource.from_string(source).value
        trends = []
        today = _now_utc().date()

        for i in range(months):
            target_month = today.month - i
            target_year = today.year
            while target_month <= 0:
                target_month += 12
                target_year -= 1

            ref = date(target_year, target_month, 1)
            start, end = _month_bounds(ref)

            payments = self.list_payments(
                source=source_val,
                date_range=(start, end),
            )
            active = [
                p for p in payments
                if p.status in (
                    PaymentStatus.RECEIVED.value,
                    PaymentStatus.CLEARED.value,
                )
            ]

            total = sum(p.net_amount for p in active)
            count = len(active)
            avg = _round_amount(total / count) if count > 0 else 0.0

            trends.append({
                "month": ref.strftime("%Y-%m"),
                "amount": _round_amount(total),
                "count": count,
                "avg": avg,
            })

        trends.reverse()
        return trends

    def get_fee_analysis(self, days: int = 30) -> dict:
        """
        Analyze fees paid across all sources over the last N days.
        Returns breakdown by source with totals and percentages.
        """
        start = (_now_utc().date() - timedelta(days=days)).isoformat()
        end = _today_iso()

        payments = self.list_payments(date_range=(start, end))
        active = [
            p for p in payments
            if p.status in (
                PaymentStatus.RECEIVED.value,
                PaymentStatus.CLEARED.value,
            )
        ]

        total_fees = 0.0
        total_gross = 0.0
        by_source: dict[str, dict] = {}

        for p in active:
            total_fees += p.fees
            total_gross += p.amount

            if p.source not in by_source:
                by_source[p.source] = {
                    "total_fees": 0.0,
                    "total_gross": 0.0,
                    "payment_count": 0,
                    "fee_rate": 0.0,
                }
            entry = by_source[p.source]
            entry["total_fees"] += p.fees
            entry["total_gross"] += p.amount
            entry["payment_count"] += 1

        # Calculate effective fee rates
        for src, entry in by_source.items():
            entry["total_fees"] = _round_amount(entry["total_fees"])
            entry["total_gross"] = _round_amount(entry["total_gross"])
            if entry["total_gross"] > 0:
                entry["fee_rate"] = _round_amount(
                    entry["total_fees"] / entry["total_gross"] * 100
                )

        overall_rate = _round_amount(
            total_fees / total_gross * 100
        ) if total_gross > 0 else 0.0

        return {
            "period": f"Last {days} days",
            "total_fees": _round_amount(total_fees),
            "total_gross": _round_amount(total_gross),
            "total_net": _round_amount(total_gross - total_fees),
            "overall_fee_rate": overall_rate,
            "by_source": dict(sorted(
                by_source.items(),
                key=lambda kv: kv[1]["total_fees"],
                reverse=True,
            )),
            "payment_count": len(active),
        }

    def get_payment_velocity(self) -> dict:
        """
        Calculate payment velocity metrics: how fast payments flow in.
        Returns avg payments/day, avg amount/day, trends.
        """
        today = _now_utc().date()

        # Last 7 days
        start_7 = (today - timedelta(days=7)).isoformat()
        # Last 30 days
        start_30 = (today - timedelta(days=30)).isoformat()
        # Last 90 days
        start_90 = (today - timedelta(days=90)).isoformat()

        end = today.isoformat()

        def velocity_for_range(start: str, days: int) -> dict:
            payments = self.list_payments(date_range=(start, end))
            active = [
                p for p in payments
                if p.status in (
                    PaymentStatus.RECEIVED.value,
                    PaymentStatus.CLEARED.value,
                )
            ]
            total = sum(p.net_amount for p in active)
            count = len(active)
            return {
                "payments_per_day": _round_amount(count / days) if days > 0 else 0,
                "amount_per_day": _round_amount(total / days) if days > 0 else 0,
                "total_amount": _round_amount(total),
                "total_count": count,
            }

        return {
            "last_7_days": velocity_for_range(start_7, 7),
            "last_30_days": velocity_for_range(start_30, 30),
            "last_90_days": velocity_for_range(start_90, 90),
            "calculated_at": _now_iso(),
        }

    # ===================================================================
    # REPORTING
    # ===================================================================

    async def generate_financial_report(self, months: int = 3) -> dict:
        """
        Generate a comprehensive financial report spanning N months.
        Includes monthly summaries, source analysis, reconciliation
        status, overdue alerts, and projections.
        """
        today = _now_utc().date()
        start = (today - timedelta(days=months * 30)).isoformat()
        end = today.isoformat()

        # Overall breakdown
        overall = self.get_revenue_breakdown(start, end)

        # Monthly summaries
        monthly = self.get_monthly_summary(months)

        # Fee analysis
        fees = self.get_fee_analysis(days=months * 30)

        # Velocity
        velocity = self.get_payment_velocity()

        # Overdue payments
        overdue = self.check_overdue_payments()
        overdue_data = [
            {
                "schedule_id": s.schedule_id,
                "source": s.source,
                "expected_amount": s.expected_amount,
                "next_expected_date": s.next_expected_date,
                "frequency": s.frequency,
            }
            for s in overdue
        ]

        # Expected income (next 30 days)
        expected = self.get_expected_income(30)
        expected_total = sum(e["amount"] for e in expected)

        # Recent discrepancies
        discrepancies = self.get_discrepancies(days=months * 30)
        disc_data = [
            {
                "record_id": d.record_id,
                "source": d.source,
                "period": f"{d.period_start} to {d.period_end}",
                "expected": d.expected_amount,
                "actual": d.actual_amount,
                "discrepancy": d.discrepancy,
            }
            for d in discrepancies[:10]
        ]

        # Source trends (top 5 by net revenue)
        top_sources = list(overall.by_source.keys())[:5]
        source_trends = {}
        for src in top_sources:
            source_trends[src] = self.get_source_trends(src, months)

        # Pending invoices
        pending_invoices = self.list_invoices(status="sent")
        pending_total = sum(i.total for i in pending_invoices)

        report = {
            "report_title": f"Financial Report — Last {months} Months",
            "generated_at": _now_iso(),
            "period": f"{start} to {end}",
            "summary": {
                "total_gross": overall.total_gross,
                "total_fees": overall.total_fees,
                "total_net": overall.total_net,
                "payment_count": overall.payment_count,
                "avg_payment": overall.avg_payment,
                "top_source": list(overall.by_source.keys())[0] if overall.by_source else "none",
                "top_site": list(overall.by_site.keys())[0] if overall.by_site else "none",
            },
            "monthly_breakdown": monthly,
            "by_source": overall.by_source,
            "by_site": overall.by_site,
            "fee_analysis": fees,
            "velocity": velocity,
            "source_trends": source_trends,
            "overdue_payments": overdue_data,
            "overdue_count": len(overdue_data),
            "discrepancies": disc_data,
            "discrepancy_count": len(disc_data),
            "pending_invoices": len(pending_invoices),
            "pending_invoice_total": _round_amount(pending_total),
            "expected_income_30d": _round_amount(expected_total),
            "expected_income_details": expected,
        }

        logger.info(
            "Generated financial report for %d months: net=%s, %d payments",
            months, _format_currency(overall.total_net), overall.payment_count,
        )
        return report

    def get_tax_summary(self, year: int) -> dict:
        """
        Generate a tax summary for a given year.
        Categorizes income by source with totals for tax reporting.
        """
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        payments = self.list_payments(date_range=(start, end))
        active = [
            p for p in payments
            if p.status in (
                PaymentStatus.RECEIVED.value,
                PaymentStatus.CLEARED.value,
            )
        ]

        # Categorize for tax purposes
        categories: dict[str, dict] = {
            "advertising_income": {
                "sources": [PaymentSource.ADSENSE.value],
                "total": 0.0,
                "count": 0,
                "payments": [],
            },
            "affiliate_income": {
                "sources": [
                    PaymentSource.AMAZON_AFFILIATE.value,
                    PaymentSource.OTHER_AFFILIATE.value,
                ],
                "total": 0.0,
                "count": 0,
                "payments": [],
            },
            "royalty_income": {
                "sources": [PaymentSource.KDP_ROYALTY.value],
                "total": 0.0,
                "count": 0,
                "payments": [],
            },
            "product_sales": {
                "sources": [
                    PaymentSource.ETSY_SALES.value,
                    PaymentSource.DIGITAL_PRODUCT.value,
                ],
                "total": 0.0,
                "count": 0,
                "payments": [],
            },
            "subscription_income": {
                "sources": [
                    PaymentSource.SUBSTACK_PAID.value,
                    PaymentSource.COURSE.value,
                ],
                "total": 0.0,
                "count": 0,
                "payments": [],
            },
            "service_income": {
                "sources": [
                    PaymentSource.SPONSORSHIP.value,
                    PaymentSource.CONSULTING.value,
                ],
                "total": 0.0,
                "count": 0,
                "payments": [],
            },
        }

        total_gross = 0.0
        total_fees = 0.0
        total_net = 0.0

        for p in active:
            total_gross += p.amount
            total_fees += p.fees
            total_net += p.net_amount

            # Assign to category
            for cat_name, cat in categories.items():
                if p.source in cat["sources"]:
                    cat["total"] += p.net_amount
                    cat["count"] += 1
                    cat["payments"].append({
                        "payment_id": p.payment_id,
                        "date": p.received_date,
                        "amount": p.amount,
                        "fees": p.fees,
                        "net": p.net_amount,
                        "source": p.source,
                        "description": p.description,
                    })
                    break

        # Round totals
        for cat in categories.values():
            cat["total"] = _round_amount(cat["total"])

        # Quarterly breakdown
        quarters = {}
        for q in range(1, 5):
            q_start = f"{year}-{(q - 1) * 3 + 1:02d}-01"
            q_end_month = q * 3
            if q_end_month == 12:
                q_end = f"{year}-12-31"
            else:
                q_end = (
                    date(year, q_end_month + 1, 1) - timedelta(days=1)
                ).isoformat()

            q_payments = [
                p for p in active
                if q_start <= p.received_date <= q_end
            ]
            quarters[f"Q{q}"] = {
                "gross": _round_amount(sum(p.amount for p in q_payments)),
                "fees": _round_amount(sum(p.fees for p in q_payments)),
                "net": _round_amount(sum(p.net_amount for p in q_payments)),
                "count": len(q_payments),
            }

        return {
            "year": year,
            "generated_at": _now_iso(),
            "total_gross": _round_amount(total_gross),
            "total_fees": _round_amount(total_fees),
            "total_net": _round_amount(total_net),
            "total_deductible_fees": _round_amount(total_fees),
            "payment_count": len(active),
            "categories": {
                k: {
                    "total": v["total"],
                    "count": v["count"],
                }
                for k, v in categories.items()
            },
            "category_details": categories,
            "quarterly": quarters,
            "sites_active": list(set(
                p.site_id for p in active if p.site_id
            )),
        }

    def get_stats(self) -> dict:
        """Get a quick overview of the payment system state."""
        today = _today_iso()
        start_30 = (_now_utc().date() - timedelta(days=30)).isoformat()

        all_payments = list(self._payments.values())
        all_invoices = list(self._invoices.values())
        all_recon = list(self._reconciliations.values())
        schedules = self.list_schedules()
        overdue = self.check_overdue_payments()

        # Payment status counts
        status_counts: dict[str, int] = {}
        for p in all_payments:
            st = p.get("status", "unknown")
            status_counts[st] = status_counts.get(st, 0) + 1

        # Source counts
        source_counts: dict[str, int] = {}
        for p in all_payments:
            src = p.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        # Recent 30-day totals
        recent = [
            p for p in all_payments
            if p.get("received_date", "") >= start_30
            and p.get("status") in (
                PaymentStatus.RECEIVED.value,
                PaymentStatus.CLEARED.value,
            )
        ]
        recent_total = _round_amount(sum(p.get("net_amount", 0) for p in recent))

        # Invoice status counts
        inv_status_counts: dict[str, int] = {}
        for inv in all_invoices:
            st = inv.get("status", "unknown")
            inv_status_counts[st] = inv_status_counts.get(st, 0) + 1

        # Outstanding invoice total
        outstanding = sum(
            inv.get("total", 0) for inv in all_invoices
            if inv.get("status") in (InvoiceStatus.SENT.value, InvoiceStatus.OVERDUE.value)
        )

        return {
            "generated_at": _now_iso(),
            "payments": {
                "total_count": len(all_payments),
                "by_status": status_counts,
                "by_source": source_counts,
                "last_30_days_net": recent_total,
                "last_30_days_count": len(recent),
            },
            "invoices": {
                "total_count": len(all_invoices),
                "by_status": inv_status_counts,
                "outstanding_total": _round_amount(outstanding),
            },
            "reconciliation": {
                "total_count": len(all_recon),
                "discrepancies": len([
                    r for r in all_recon
                    if r.get("status") == ReconciliationStatus.DISCREPANCY.value
                ]),
            },
            "schedules": {
                "total_count": len(schedules),
                "active_count": len([s for s in schedules if s.active]),
                "overdue_count": len(overdue),
            },
            "data_files": {
                "payments": str(PAYMENTS_FILE),
                "invoices": str(INVOICES_FILE),
                "reconciliation": str(RECONCILIATION_FILE),
                "config": str(CONFIG_FILE),
            },
        }


# ===========================================================================
# Singleton Access
# ===========================================================================

_processor_instance: Optional[PaymentProcessor] = None


def get_processor() -> PaymentProcessor:
    """Return the singleton PaymentProcessor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = PaymentProcessor()
    return _processor_instance


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def _cli_main() -> None:
    """CLI entry point: python -m src.payment_processor <command> [options]."""

    parser = argparse.ArgumentParser(
        prog="payment_processor",
        description="Payment Processor -- OpenClaw Empire CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- record ---
    p_rec = sub.add_parser("record", help="Record a new payment")
    p_rec.add_argument("--source", required=True, help="Payment source (e.g. adsense, kdp_royalty)")
    p_rec.add_argument("--amount", required=True, type=float, help="Payment amount")
    p_rec.add_argument("--currency", default=DEFAULT_CURRENCY, help="Currency (default: USD)")
    p_rec.add_argument("--status", default="received", help="Payment status")
    p_rec.add_argument("--site", dest="site_id", default="", help="Site ID")
    p_rec.add_argument("--description", default="", help="Description")
    p_rec.add_argument("--reference", dest="reference_id", default="", help="External reference ID")
    p_rec.add_argument("--date", dest="received_date", default="", help="Received date (YYYY-MM-DD)")
    p_rec.add_argument("--fees", type=float, default=0.0, help="Fees deducted")
    p_rec.add_argument("--method", dest="payment_method", default="", help="Payment method")

    # --- payments ---
    p_pay = sub.add_parser("payments", help="List payments")
    p_pay.add_argument("--source", default=None, help="Filter by source")
    p_pay.add_argument("--status", default=None, help="Filter by status")
    p_pay.add_argument("--site", dest="site_id", default=None, help="Filter by site")
    p_pay.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p_pay.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p_pay.add_argument("--limit", type=int, default=25, help="Max results")

    # --- import ---
    p_imp = sub.add_parser("import", help="Import payments from CSV/JSON file")
    p_imp.add_argument("--source", required=True, help="Payment source")
    p_imp.add_argument("--file", required=True, help="Path to CSV or JSON file")

    # --- invoices ---
    p_inv = sub.add_parser("invoices", help="List invoices")
    p_inv.add_argument("--status", default=None, help="Filter by status")
    p_inv.add_argument("--client", default=None, help="Filter by client name")
    p_inv.add_argument("--limit", type=int, default=25, help="Max results")

    # --- create-invoice ---
    p_cinv = sub.add_parser("create-invoice", help="Create a new invoice")
    p_cinv.add_argument("--client", required=True, help="Client name")
    p_cinv.add_argument("--email", required=True, help="Client email")
    p_cinv.add_argument(
        "--items", required=True,
        help='JSON array of items: [{"description":"...", "quantity":1, "unit_price":100}]',
    )
    p_cinv.add_argument("--currency", default=DEFAULT_CURRENCY, help="Currency")
    p_cinv.add_argument("--tax-rate", type=float, default=0.0, help="Tax rate (e.g. 0.07 for 7%%)")
    p_cinv.add_argument("--notes", default="", help="Invoice notes")
    p_cinv.add_argument("--due-days", type=int, default=30, help="Days until due")

    # --- reconcile ---
    p_recon = sub.add_parser("reconcile", help="Reconcile expected vs actual revenue")
    p_recon.add_argument("--source", required=True, help="Payment source")
    p_recon.add_argument("--start", required=True, help="Period start date (YYYY-MM-DD)")
    p_recon.add_argument("--end", required=True, help="Period end date (YYYY-MM-DD)")
    p_recon.add_argument("--expected", required=True, type=float, help="Expected amount")

    # --- reconciliations ---
    p_recs = sub.add_parser("reconciliations", help="List reconciliation records")
    p_recs.add_argument("--source", default=None, help="Filter by source")
    p_recs.add_argument("--status", default=None, help="Filter by status")
    p_recs.add_argument("--limit", type=int, default=25, help="Max results")

    # --- schedules ---
    p_sched = sub.add_parser("schedules", help="List payment schedules")
    p_sched.add_argument("--active", action="store_true", help="Active only")

    # --- create-schedule ---
    p_csched = sub.add_parser("create-schedule", help="Create a payment schedule")
    p_csched.add_argument("--source", required=True, help="Payment source")
    p_csched.add_argument("--amount", required=True, type=float, help="Expected amount")
    p_csched.add_argument("--frequency", default="monthly", help="Payment frequency")
    p_csched.add_argument("--next-date", default="", help="Next expected date (YYYY-MM-DD)")

    # --- overdue ---
    sub.add_parser("overdue", help="Check for overdue payments")

    # --- expected ---
    p_exp = sub.add_parser("expected", help="Show expected income")
    p_exp.add_argument("--days", type=int, default=30, help="Days ahead to project")

    # --- breakdown ---
    p_brk = sub.add_parser("breakdown", help="Revenue breakdown for a period")
    p_brk.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_brk.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

    # --- monthly ---
    p_mon = sub.add_parser("monthly", help="Monthly revenue summary")
    p_mon.add_argument("--months", type=int, default=12, help="Number of months")

    # --- fees ---
    p_fees = sub.add_parser("fees", help="Fee analysis")
    p_fees.add_argument("--days", type=int, default=30, help="Number of days")

    # --- tax ---
    p_tax = sub.add_parser("tax", help="Tax summary for a year")
    p_tax.add_argument("--year", required=True, type=int, help="Tax year")

    # --- report ---
    p_rep = sub.add_parser("report", help="Generate financial report")
    p_rep.add_argument("--months", type=int, default=3, help="Number of months")

    # --- stats ---
    sub.add_parser("stats", help="Quick system stats")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    proc = get_processor()

    try:
        _dispatch_cli(args, proc)
    except (KeyError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\nFatal: {exc}", file=sys.stderr)
        sys.exit(2)


def _dispatch_cli(args: argparse.Namespace, proc: PaymentProcessor) -> None:
    """Dispatch CLI command to the appropriate handler."""

    if args.command == "record":
        payment = proc.record_payment(
            source=args.source,
            amount=args.amount,
            currency=args.currency,
            status=args.status,
            reference_id=args.reference_id,
            site_id=args.site_id,
            description=args.description,
            received_date=args.received_date,
            fees=args.fees,
            payment_method=args.payment_method,
        )
        print(f"\nPayment recorded:")
        print(f"  ID:          {payment.payment_id}")
        print(f"  Source:      {payment.source}")
        print(f"  Amount:      {_format_currency(payment.amount, payment.currency)}")
        print(f"  Fees:        {_format_currency(payment.fees, payment.currency)}")
        print(f"  Net:         {_format_currency(payment.net_amount, payment.currency)}")
        print(f"  Status:      {payment.status}")
        print(f"  Site:        {payment.site_id or '(general)'}")
        print(f"  Received:    {payment.received_date}")

    elif args.command == "payments":
        date_range = None
        if args.start and args.end:
            date_range = (args.start, args.end)

        payments = proc.list_payments(
            source=args.source,
            status=args.status,
            site_id=args.site_id,
            date_range=date_range,
            limit=args.limit,
        )
        if not payments:
            print("No payments found.")
            return

        print(f"\nPAYMENTS ({len(payments)})")
        print(f"{'=' * 90}")
        print(
            f"  {'Date':<12s} {'Source':<18s} {'Status':<10s} "
            f"{'Site':<16s} {'Amount':>10s} {'Net':>10s}"
        )
        print(f"  {'-' * 12} {'-' * 18} {'-' * 10} {'-' * 16} {'-' * 10} {'-' * 10}")

        total_net = 0.0
        for p in payments:
            total_net += p.net_amount
            print(
                f"  {p.received_date:<12s} {p.source:<18s} {p.status:<10s} "
                f"{(p.site_id or '-'):<16s} "
                f"{_format_currency(p.amount, p.currency):>10s} "
                f"{_format_currency(p.net_amount, p.currency):>10s}"
            )

        print(f"\n  Total net: {_format_currency(total_net)}")

    elif args.command == "import":
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        data = []
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    data = data.get("payments", data.get("data", [data]))
        elif suffix == ".csv":
            import csv
            with open(file_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                data = list(reader)
        else:
            print(f"Unsupported file format: {suffix}. Use .json or .csv.", file=sys.stderr)
            sys.exit(1)

        imported = _run_sync(proc.import_payments(args.source, data))
        print(f"\nImported {len(imported)} payments from {file_path.name}")
        total = sum(p.amount for p in imported)
        print(f"  Total amount: {_format_currency(total)}")

    elif args.command == "invoices":
        invoices = proc.list_invoices(
            status=args.status,
            client_name=args.client,
            limit=args.limit,
        )
        if not invoices:
            print("No invoices found.")
            return

        print(f"\nINVOICES ({len(invoices)})")
        print(f"{'=' * 85}")
        print(
            f"  {'Number':<16s} {'Client':<20s} {'Status':<10s} "
            f"{'Issued':<12s} {'Due':<12s} {'Total':>10s}"
        )
        print(f"  {'-' * 16} {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}")

        for inv in invoices:
            inv_number = inv.metadata.get("invoice_number", inv.invoice_id[:16])
            print(
                f"  {inv_number:<16s} {inv.client_name[:20]:<20s} "
                f"{inv.status:<10s} {inv.issued_date:<12s} {inv.due_date:<12s} "
                f"{_format_currency(inv.total, inv.currency):>10s}"
            )

    elif args.command == "create-invoice":
        try:
            items = json.loads(args.items)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON for --items: {e}", file=sys.stderr)
            sys.exit(1)

        inv = proc.create_invoice(
            client_name=args.client,
            client_email=args.email,
            items=items,
            currency=args.currency,
            tax_rate=args.tax_rate,
            notes=args.notes,
            due_days=args.due_days,
        )
        inv_number = inv.metadata.get("invoice_number", inv.invoice_id)
        print(f"\nInvoice created:")
        print(f"  ID:       {inv.invoice_id}")
        print(f"  Number:   {inv_number}")
        print(f"  Client:   {inv.client_name}")
        print(f"  Items:    {len(inv.items)}")
        print(f"  Subtotal: {_format_currency(inv.subtotal, inv.currency)}")
        print(f"  Tax:      {_format_currency(inv.tax_amount, inv.currency)}")
        print(f"  Total:    {_format_currency(inv.total, inv.currency)}")
        print(f"  Due:      {inv.due_date}")

    elif args.command == "reconcile":
        record = _run_sync(proc.reconcile(
            source=args.source,
            period_start=args.start,
            period_end=args.end,
            expected_amount=args.expected,
        ))
        print(f"\nReconciliation result:")
        print(f"  ID:          {record.record_id}")
        print(f"  Source:      {record.source}")
        print(f"  Period:      {record.period_start} to {record.period_end}")
        print(f"  Expected:    {_format_currency(record.expected_amount)}")
        print(f"  Actual:      {_format_currency(record.actual_amount)}")
        print(f"  Discrepancy: {_format_currency(record.discrepancy)}")
        print(f"  Status:      {record.status}")
        print(f"  Matched:     {len(record.payments_matched)} payments")
        print(f"  Unmatched:   {len(record.payments_unmatched)} payments")
        print(f"  Notes:       {record.notes}")

    elif args.command == "reconciliations":
        records = proc.list_reconciliations(
            source=args.source,
            status=args.status,
            limit=args.limit,
        )
        if not records:
            print("No reconciliation records found.")
            return

        print(f"\nRECONCILIATIONS ({len(records)})")
        print(f"{'=' * 90}")
        print(
            f"  {'Source':<18s} {'Period':<24s} {'Status':<12s} "
            f"{'Expected':>10s} {'Actual':>10s} {'Disc':>10s}"
        )
        print(f"  {'-' * 18} {'-' * 24} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

        for r in records:
            period = f"{r.period_start} to {r.period_end}"
            print(
                f"  {r.source:<18s} {period:<24s} {r.status:<12s} "
                f"{_format_currency(r.expected_amount):>10s} "
                f"{_format_currency(r.actual_amount):>10s} "
                f"{_format_currency(r.discrepancy):>10s}"
            )

    elif args.command == "schedules":
        schedules = proc.list_schedules(active_only=getattr(args, "active", False))
        if not schedules:
            print("No payment schedules found.")
            return

        print(f"\nPAYMENT SCHEDULES ({len(schedules)})")
        print(f"{'=' * 80}")
        print(
            f"  {'Source':<18s} {'Amount':>10s} {'Frequency':<12s} "
            f"{'Next Date':<12s} {'Active':<8s}"
        )
        print(f"  {'-' * 18} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 8}")

        for s in schedules:
            print(
                f"  {s.source:<18s} "
                f"{_format_currency(s.expected_amount):>10s} "
                f"{s.frequency:<12s} {s.next_expected_date:<12s} "
                f"{'Yes' if s.active else 'No':<8s}"
            )

    elif args.command == "create-schedule":
        sched = proc.create_schedule(
            source=args.source,
            expected_amount=args.amount,
            frequency=args.frequency,
            next_expected_date=args.next_date,
        )
        print(f"\nSchedule created:")
        print(f"  ID:        {sched.schedule_id}")
        print(f"  Source:    {sched.source}")
        print(f"  Amount:    {_format_currency(sched.expected_amount)}")
        print(f"  Frequency: {sched.frequency}")
        print(f"  Next:      {sched.next_expected_date}")

    elif args.command == "overdue":
        overdue = proc.check_overdue_payments()
        if not overdue:
            print("\nNo overdue payments found. All schedules on track.")
            return

        print(f"\nOVERDUE PAYMENTS ({len(overdue)})")
        print(f"{'=' * 70}")
        for s in overdue:
            days_late = _days_between(s.next_expected_date, _today_iso())
            print(f"  {s.source:<18s} {_format_currency(s.expected_amount):>10s} "
                  f"  due: {s.next_expected_date}  ({days_late} days late)")

    elif args.command == "expected":
        projections = proc.get_expected_income(args.days)
        if not projections:
            print(f"\nNo expected income in the next {args.days} days.")
            return

        total = sum(p["amount"] for p in projections)
        print(f"\nEXPECTED INCOME — Next {args.days} Days")
        print(f"{'=' * 60}")
        print(f"  {'Date':<12s} {'Source':<18s} {'Amount':>10s} {'Frequency':<12s}")
        print(f"  {'-' * 12} {'-' * 18} {'-' * 10} {'-' * 12}")

        for p in projections:
            print(
                f"  {p['expected_date']:<12s} {p['source']:<18s} "
                f"{_format_currency(p['amount']):>10s} {p['frequency']:<12s}"
            )
        print(f"\n  Total expected: {_format_currency(total)}")

    elif args.command == "breakdown":
        breakdown = proc.get_revenue_breakdown(args.start, args.end)
        print(f"\nREVENUE BREAKDOWN: {breakdown.period}")
        print(f"{'=' * 50}")
        print(f"  Gross:       {_format_currency(breakdown.total_gross)}")
        print(f"  Fees:        {_format_currency(breakdown.total_fees)}")
        print(f"  Net:         {_format_currency(breakdown.total_net)}")
        print(f"  Payments:    {breakdown.payment_count}")
        print(f"  Avg Payment: {_format_currency(breakdown.avg_payment)}")

        if breakdown.by_source:
            print(f"\n  By Source:")
            for src, amt in breakdown.by_source.items():
                print(f"    {src:<20s} {_format_currency(amt):>10s}")

        if breakdown.by_site:
            print(f"\n  By Site:")
            for site, amt in breakdown.by_site.items():
                print(f"    {site:<20s} {_format_currency(amt):>10s}")

    elif args.command == "monthly":
        summaries = proc.get_monthly_summary(args.months)
        if not summaries:
            print("No monthly data available.")
            return

        print(f"\nMONTHLY REVENUE SUMMARY (Last {args.months} Months)")
        print(f"{'=' * 65}")
        print(f"  {'Month':<10s} {'Gross':>10s} {'Fees':>10s} "
              f"{'Net':>10s} {'Count':>8s}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")

        for m in summaries:
            print(
                f"  {m['month']:<10s} "
                f"{_format_currency(m['gross']):>10s} "
                f"{_format_currency(m['fees']):>10s} "
                f"{_format_currency(m['net']):>10s} "
                f"{m['count']:>8d}"
            )

        total_net = sum(m["net"] for m in summaries)
        total_count = sum(m["count"] for m in summaries)
        print(f"\n  Total: {_format_currency(total_net)} net across {total_count} payments")

    elif args.command == "fees":
        analysis = proc.get_fee_analysis(args.days)
        print(f"\nFEE ANALYSIS — {analysis['period']}")
        print(f"{'=' * 55}")
        print(f"  Gross Revenue: {_format_currency(analysis['total_gross'])}")
        print(f"  Total Fees:    {_format_currency(analysis['total_fees'])}")
        print(f"  Net Revenue:   {_format_currency(analysis['total_net'])}")
        print(f"  Fee Rate:      {analysis['overall_fee_rate']}%")
        print(f"  Payments:      {analysis['payment_count']}")

        if analysis["by_source"]:
            print(f"\n  By Source:")
            print(f"    {'Source':<18s} {'Fees':>10s} {'Rate':>8s} {'Count':>8s}")
            print(f"    {'-' * 18} {'-' * 10} {'-' * 8} {'-' * 8}")
            for src, data in analysis["by_source"].items():
                print(
                    f"    {src:<18s} "
                    f"{_format_currency(data['total_fees']):>10s} "
                    f"{data['fee_rate']:>7.1f}% "
                    f"{data['payment_count']:>8d}"
                )

    elif args.command == "tax":
        summary = proc.get_tax_summary(args.year)
        print(f"\nTAX SUMMARY — {summary['year']}")
        print(f"{'=' * 50}")
        print(f"  Gross Income:      {_format_currency(summary['total_gross'])}")
        print(f"  Deductible Fees:   {_format_currency(summary['total_fees'])}")
        print(f"  Net Income:        {_format_currency(summary['total_net'])}")
        print(f"  Payments:          {summary['payment_count']}")

        if summary.get("categories"):
            print(f"\n  Income Categories:")
            for cat, data in summary["categories"].items():
                label = cat.replace("_", " ").title()
                if data["total"] > 0:
                    print(f"    {label:<25s} {_format_currency(data['total']):>10s} ({data['count']} payments)")

        if summary.get("quarterly"):
            print(f"\n  Quarterly Breakdown:")
            for q, data in summary["quarterly"].items():
                print(f"    {q}: Net {_format_currency(data['net'])} ({data['count']} payments)")

        if summary.get("sites_active"):
            print(f"\n  Active Sites: {', '.join(summary['sites_active'])}")

    elif args.command == "report":
        report = _run_sync(proc.generate_financial_report(args.months))
        print(f"\n{report['report_title']}")
        print(f"{'=' * 60}")
        print(f"  Period: {report['period']}")
        print(f"  Generated: {report['generated_at']}")

        s = report["summary"]
        print(f"\n  SUMMARY")
        print(f"  {'=' * 40}")
        print(f"  Gross Revenue:    {_format_currency(s['total_gross'])}")
        print(f"  Total Fees:       {_format_currency(s['total_fees'])}")
        print(f"  Net Revenue:      {_format_currency(s['total_net'])}")
        print(f"  Payments:         {s['payment_count']}")
        print(f"  Avg Payment:      {_format_currency(s['avg_payment'])}")
        print(f"  Top Source:       {s['top_source']}")
        print(f"  Top Site:         {s['top_site']}")

        if report.get("overdue_count", 0) > 0:
            print(f"\n  OVERDUE: {report['overdue_count']} payment(s) overdue")
            for o in report["overdue_payments"]:
                print(f"    - {o['source']}: {_format_currency(o['expected_amount'])} "
                      f"(due {o['next_expected_date']})")

        if report.get("discrepancy_count", 0) > 0:
            print(f"\n  DISCREPANCIES: {report['discrepancy_count']} found")
            for d in report["discrepancies"]:
                print(f"    - {d['source']} ({d['period']}): "
                      f"expected {_format_currency(d['expected'])} "
                      f"got {_format_currency(d['actual'])} "
                      f"(diff: {_format_currency(d['discrepancy'])})")

        print(f"\n  Expected Income (30d):  {_format_currency(report['expected_income_30d'])}")
        print(f"  Pending Invoices:       {report['pending_invoices']} "
              f"({_format_currency(report['pending_invoice_total'])})")

        # Monthly trend
        if report.get("monthly_breakdown"):
            print(f"\n  MONTHLY TREND")
            print(f"  {'Month':<10s} {'Net':>10s} {'Count':>8s}")
            print(f"  {'-' * 10} {'-' * 10} {'-' * 8}")
            for m in report["monthly_breakdown"]:
                print(f"  {m['month']:<10s} {_format_currency(m['net']):>10s} {m['count']:>8d}")

    elif args.command == "stats":
        stats = proc.get_stats()
        print(f"\nPAYMENT PROCESSOR STATS")
        print(f"{'=' * 50}")

        ps = stats["payments"]
        print(f"\n  Payments:")
        print(f"    Total:        {ps['total_count']}")
        print(f"    Last 30 days: {ps['last_30_days_count']} "
              f"(net: {_format_currency(ps['last_30_days_net'])})")
        if ps.get("by_status"):
            for st, cnt in ps["by_status"].items():
                print(f"    [{st}]: {cnt}")

        inv = stats["invoices"]
        print(f"\n  Invoices:")
        print(f"    Total:        {inv['total_count']}")
        print(f"    Outstanding:  {_format_currency(inv['outstanding_total'])}")

        rec = stats["reconciliation"]
        print(f"\n  Reconciliation:")
        print(f"    Total:         {rec['total_count']}")
        print(f"    Discrepancies: {rec['discrepancies']}")

        sch = stats["schedules"]
        print(f"\n  Schedules:")
        print(f"    Total:   {sch['total_count']}")
        print(f"    Active:  {sch['active_count']}")
        print(f"    Overdue: {sch['overdue_count']}")

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Module entry point."""
    _cli_main()


if __name__ == "__main__":
    main()
