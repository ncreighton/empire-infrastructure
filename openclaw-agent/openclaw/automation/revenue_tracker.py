"""RevenueTracker — track earnings and sales across all platforms.

Monitors revenue from each platform where OpenClaw has an active account.
Stores revenue events in the Codex database and provides aggregated reports.

Revenue sources:
- Direct product sales (Gumroad, Etsy, Creative Market, Envato, etc.)
- Tips/donations (Ko-fi, Buy Me a Coffee)
- Affiliate commissions (linked products)
- Subscription revenue (Patreon, Teachable, Skillshare)

The tracker works in two modes:
1. **Passive**: Record revenue events from webhook callbacks (n8n, Zapier)
2. **Active**: Scrape earnings dashboards via browser agent (periodic cron)

Usage::

    tracker = RevenueTracker(codex=codex)

    # Record a sale
    tracker.record_sale("gumroad", amount=9.99, product="AI Prompt Pack",
                        currency="USD", source="webhook")

    # Get revenue report
    report = tracker.get_report(days=30)
    # → {"total_revenue": 149.85, "by_platform": {...}, "by_product": {...}}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RevenueEvent:
    """A single revenue event (sale, tip, subscription payment)."""

    platform_id: str
    amount: float
    currency: str = "USD"
    product: str = ""
    event_type: str = "sale"  # sale, tip, subscription, affiliate, refund
    source: str = "manual"   # webhook, scrape, manual
    external_id: str = ""    # platform-specific transaction ID
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


# Platform-specific earnings dashboard URLs (for active scraping)
_DASHBOARD_URLS: dict[str, str] = {
    "gumroad": "https://app.gumroad.com/dashboard",
    "etsy": "https://www.etsy.com/your/shops/me/dashboard",
    "creative_market": "https://creativemarket.com/dashboard",
    "envato": "https://account.envato.com/statements",
    "promptbase": "https://promptbase.com/dashboard",
    "kofi": "https://ko-fi.com/manage/shop-orders",
    "buymeacoffee": "https://www.buymeacoffee.com/dashboard",
    "payhip": "https://payhip.com/dashboard",
    "teachable": "https://sso.teachable.com/secure/dashboard",
    "udemy": "https://www.udemy.com/instructor/revenue/overview",
    "skillshare": "https://www.skillshare.com/teach/stats",
    "thrivecart": "https://thrivecart.com/account/stats",
    "n8n_creator_hub": "https://n8n.io/workflows",
}


class RevenueTracker:
    """Track and report revenue across all platforms.

    Stores revenue events in the PlatformCodex SQLite database and provides
    aggregation, trend analysis, and platform comparison reports.
    """

    def __init__(self, codex: Any = None, db_path: str | None = None):
        self._codex = codex
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create revenue_events table if it doesn't exist."""
        if not self._codex:
            return
        try:
            db = getattr(self._codex, "_db", None) or getattr(self._codex, "db", None)
            if db is None:
                return
            db.execute("""
                CREATE TABLE IF NOT EXISTS revenue_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    product TEXT DEFAULT '',
                    event_type TEXT DEFAULT 'sale',
                    source TEXT DEFAULT 'manual',
                    external_id TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_revenue_platform
                ON revenue_events(platform_id)
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_revenue_timestamp
                ON revenue_events(timestamp)
            """)
            db.commit()
            logger.debug("Revenue events table ready")
        except Exception as e:
            logger.debug(f"Could not create revenue table: {e}")

    def record_sale(
        self,
        platform_id: str,
        amount: float,
        product: str = "",
        currency: str = "USD",
        event_type: str = "sale",
        source: str = "manual",
        external_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RevenueEvent:
        """Record a revenue event.

        Args:
            platform_id: Platform where the sale occurred
            amount: Revenue amount (negative for refunds)
            product: Product/item name
            currency: Currency code (default USD)
            event_type: sale, tip, subscription, affiliate, refund
            source: How the event was captured (webhook, scrape, manual)
            external_id: Platform-specific transaction ID for dedup
            metadata: Additional data (buyer info, etc.)

        Returns:
            The created RevenueEvent
        """
        event = RevenueEvent(
            platform_id=platform_id,
            amount=amount,
            currency=currency,
            product=product,
            event_type=event_type,
            source=source,
            external_id=external_id,
            metadata=metadata or {},
        )

        # Dedup by external_id
        if external_id and self._event_exists(external_id):
            logger.debug(f"Revenue event {external_id} already recorded, skipping")
            return event

        self._store_event(event)
        logger.info(
            f"[{platform_id}] Revenue: ${amount:.2f} {currency} "
            f"({event_type}) — {product or 'unnamed'}"
        )
        return event

    def _event_exists(self, external_id: str) -> bool:
        """Check if an event with this external_id already exists."""
        if not self._codex:
            return False
        try:
            db = getattr(self._codex, "_db", None) or getattr(self._codex, "db", None)
            if db is None:
                return False
            cursor = db.execute(
                "SELECT 1 FROM revenue_events WHERE external_id = ? LIMIT 1",
                (external_id,),
            )
            return cursor.fetchone() is not None
        except Exception:
            return False

    def _store_event(self, event: RevenueEvent) -> None:
        """Persist a revenue event to SQLite."""
        if not self._codex:
            return
        try:
            import json
            db = getattr(self._codex, "_db", None) or getattr(self._codex, "db", None)
            if db is None:
                return
            db.execute(
                """INSERT INTO revenue_events
                   (platform_id, amount, currency, product, event_type,
                    source, external_id, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.platform_id,
                    event.amount,
                    event.currency,
                    event.product,
                    event.event_type,
                    event.source,
                    event.external_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.metadata),
                ),
            )
            db.commit()
        except Exception as e:
            logger.debug(f"Could not store revenue event: {e}")

    def get_report(self, days: int = 30) -> dict[str, Any]:
        """Generate a revenue report for the given period.

        Returns:
            Dict with total_revenue, by_platform, by_product, by_type,
            top_products, daily_trend.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        events = self._query_events(cutoff)

        # Aggregate
        total = 0.0
        by_platform: dict[str, float] = {}
        by_product: dict[str, float] = {}
        by_type: dict[str, float] = {}
        daily: dict[str, float] = {}

        for e in events:
            total += e["amount"]
            pid = e["platform_id"]
            by_platform[pid] = by_platform.get(pid, 0) + e["amount"]
            if e["product"]:
                by_product[e["product"]] = by_product.get(e["product"], 0) + e["amount"]
            etype = e["event_type"]
            by_type[etype] = by_type.get(etype, 0) + e["amount"]
            day = e["timestamp"][:10]
            daily[day] = daily.get(day, 0) + e["amount"]

        # Top products
        top_products = sorted(
            by_product.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Top platforms
        top_platforms = sorted(
            by_platform.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "period_days": days,
            "total_revenue": round(total, 2),
            "total_events": len(events),
            "by_platform": {k: round(v, 2) for k, v in top_platforms},
            "by_product": {k: round(v, 2) for k, v in top_products},
            "by_type": {k: round(v, 2) for k, v in by_type.items()},
            "daily_trend": {k: round(v, 2) for k, v in sorted(daily.items())},
            "avg_per_day": round(total / max(days, 1), 2),
        }

    def _query_events(self, since_iso: str) -> list[dict]:
        """Query revenue events since a given ISO timestamp."""
        if not self._codex:
            return []
        try:
            db = getattr(self._codex, "_db", None) or getattr(self._codex, "db", None)
            if db is None:
                return []
            cursor = db.execute(
                """SELECT platform_id, amount, currency, product, event_type,
                          source, external_id, timestamp, metadata
                   FROM revenue_events
                   WHERE timestamp >= ?
                   ORDER BY timestamp DESC""",
                (since_iso,),
            )
            rows = cursor.fetchall()
            columns = [
                "platform_id", "amount", "currency", "product", "event_type",
                "source", "external_id", "timestamp", "metadata",
            ]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.debug(f"Could not query revenue events: {e}")
            return []

    def get_platform_revenue(self, platform_id: str, days: int = 30) -> dict[str, Any]:
        """Get revenue breakdown for a single platform."""
        report = self.get_report(days)
        platform_total = report["by_platform"].get(platform_id, 0)
        return {
            "platform_id": platform_id,
            "period_days": days,
            "total_revenue": platform_total,
            "all_platforms_total": report["total_revenue"],
            "share_pct": round(
                (platform_total / report["total_revenue"] * 100)
                if report["total_revenue"] > 0
                else 0,
                1,
            ),
        }

    def get_dashboard_url(self, platform_id: str) -> str | None:
        """Get the earnings dashboard URL for a platform."""
        return _DASHBOARD_URLS.get(platform_id)

    @staticmethod
    def supported_platforms() -> list[str]:
        """Return platform IDs that have known dashboard URLs."""
        return list(_DASHBOARD_URLS.keys())
