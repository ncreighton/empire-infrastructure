"""
Buy Me a Coffee — Supporter Notifications
Sends alerts to the empire dashboard when BMC events occur.
"""
import httpx
import logging
from datetime import datetime

from bmc_config import DASHBOARD_ALERTS_ENDPOINT

logger = logging.getLogger(__name__)


async def send_dashboard_alert(
    title: str,
    message: str,
    severity: str = "info",
    source: str = "bmc-webhook",
) -> bool:
    """Send an alert to the empire dashboard."""
    payload = {
        "title": title,
        "message": message,
        "severity": severity,
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(DASHBOARD_ALERTS_ENDPOINT, json=payload)
            if resp.status_code < 300:
                logger.info(f"Dashboard alert sent: {title}")
                return True
            else:
                logger.warning(f"Dashboard alert failed ({resp.status_code}): {resp.text}")
                return False
    except Exception as e:
        logger.error(f"Failed to send dashboard alert: {e}")
        return False


async def notify_tip(supporter_name: str, amount: float, potions: int, message: str | None = None):
    """Notify dashboard about a new tip."""
    msg = f"{supporter_name} bought {potions} potion{'s' if potions != 1 else ''} (${amount:.2f})"
    if message:
        msg += f' — "{message}"'
    await send_dashboard_alert(
        title=f"BMC Tip: ${amount:.2f}",
        message=msg,
        severity="info",
    )


async def notify_shop_purchase(supporter_name: str, product_title: str, amount: float):
    """Notify dashboard about a shop purchase."""
    await send_dashboard_alert(
        title=f"BMC Sale: {product_title}",
        message=f"{supporter_name} purchased '{product_title}' for ${amount:.2f}",
        severity="info",
    )


async def notify_membership_started(supporter_name: str, tier_name: str, amount: float):
    """Notify dashboard about a new membership."""
    await send_dashboard_alert(
        title=f"New Member: {tier_name}",
        message=f"{supporter_name} joined the {tier_name} (${amount:.2f}/mo)",
        severity="info",
    )


async def notify_membership_cancelled(supporter_name: str, tier_name: str):
    """Notify dashboard about a membership cancellation (retention alert)."""
    await send_dashboard_alert(
        title=f"Member Lost: {tier_name}",
        message=f"{supporter_name} cancelled their {tier_name} membership. Consider a retention outreach.",
        severity="warning",
    )
