"""
Buy Me a Coffee — Webhook Handler
FastAPI service on port 8095 that receives BMC webhooks,
logs supporter activity, and sends alerts to the empire dashboard.

Start:
    cd bmc-witchcraft/automation
    PYTHONPATH=. python -m uvicorn bmc_webhook_handler:app --port 8095

BMC Webhook Setup:
    1. Go to BMC Dashboard → Settings → Webhooks
    2. Set URL: https://<your-domain>:8095/webhooks/bmc
    3. Copy the signing secret into BMC_WEBHOOK_SECRET env var
"""
import hmac
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel

from bmc_config import (
    BMC_WEBHOOK_SECRET,
    WEBHOOK_PORT,
    DATA_DIR,
    SUPPORTERS_LOG,
    STATS_FILE,
    TIER_MAP,
)
from supporter_notifications import (
    notify_tip,
    notify_shop_purchase,
    notify_membership_started,
    notify_membership_cancelled,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("bmc-webhook")

app = FastAPI(
    title="BMC Webhook Handler",
    description="Buy Me a Coffee webhook receiver for Witchcraft For Beginners",
    version="1.0.0",
)


# --- Data Persistence ---

def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _append_supporter_event(event_type: str, data: dict):
    """Append a supporter event to the log file."""
    log = _load_json(SUPPORTERS_LOG)
    events = log.get("events", [])
    events.append({
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    })
    # Keep last 1000 events
    log["events"] = events[-1000:]
    _save_json(SUPPORTERS_LOG, log)


def _update_stats(event_type: str, amount: float = 0):
    """Update aggregate stats."""
    stats = _load_json(STATS_FILE)

    if "totals" not in stats:
        stats["totals"] = {
            "tips_count": 0,
            "tips_revenue": 0,
            "shop_count": 0,
            "shop_revenue": 0,
            "memberships_started": 0,
            "memberships_cancelled": 0,
            "membership_revenue": 0,
        }

    totals = stats["totals"]

    if event_type == "tip":
        totals["tips_count"] += 1
        totals["tips_revenue"] += amount
    elif event_type == "shop":
        totals["shop_count"] += 1
        totals["shop_revenue"] += amount
    elif event_type == "membership_started":
        totals["memberships_started"] += 1
        totals["membership_revenue"] += amount
    elif event_type == "membership_cancelled":
        totals["memberships_cancelled"] += 1

    stats["totals"] = totals
    stats["last_updated"] = datetime.now().isoformat()
    _save_json(STATS_FILE, stats)


# --- Signature Verification ---

def verify_signature(payload_body: bytes, signature: str) -> bool:
    """Verify BMC webhook HMAC-SHA256 signature."""
    if not BMC_WEBHOOK_SECRET:
        logger.warning("BMC_WEBHOOK_SECRET not set — skipping signature verification")
        return True

    expected = hmac.new(
        BMC_WEBHOOK_SECRET.encode(),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# --- Webhook Endpoint ---

@app.post("/webhooks/bmc")
async def handle_bmc_webhook(
    request: Request,
    x_bmc_signature: str = Header(None, alias="X-BMC-Signature"),
):
    """
    Receive and process Buy Me a Coffee webhooks.

    BMC sends these event types:
    - payment.completed (tips and shop purchases)
    - membership.started
    - membership.cancelled
    """
    body = await request.body()

    # Verify signature
    if x_bmc_signature and not verify_signature(body, x_bmc_signature):
        logger.warning("Invalid webhook signature — rejecting")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("type", "unknown")
    data = payload.get("data", {})

    logger.info(f"Received BMC webhook: {event_type}")

    # Route to handler
    if event_type == "payment.completed":
        await _handle_payment(data)
    elif event_type == "membership.started":
        await _handle_membership_started(data)
    elif event_type == "membership.cancelled":
        await _handle_membership_cancelled(data)
    else:
        logger.info(f"Unhandled BMC event type: {event_type}")

    return {"status": "ok", "event": event_type}


async def _handle_payment(data: dict):
    """Handle payment.completed — could be a tip or shop purchase."""
    supporter_name = data.get("supporter_name", "Anonymous")
    amount = float(data.get("amount", 0))
    message = data.get("message")
    is_membership = data.get("is_membership_payment", False)

    # Check if it's a shop purchase (has product info)
    product = data.get("product")

    if product:
        product_title = product.get("title", "Unknown Product")
        logger.info(f"Shop purchase: {supporter_name} bought '{product_title}' for ${amount:.2f}")
        _append_supporter_event("shop", {
            "name": supporter_name,
            "product": product_title,
            "amount": amount,
        })
        _update_stats("shop", amount)
        await notify_shop_purchase(supporter_name, product_title, amount)

    elif not is_membership:
        # It's a tip
        potions = max(1, int(amount / 3))  # $3 per potion
        logger.info(f"Tip: {supporter_name} bought {potions} potion(s) (${amount:.2f})")
        _append_supporter_event("tip", {
            "name": supporter_name,
            "amount": amount,
            "potions": potions,
            "message": message,
        })
        _update_stats("tip", amount)
        await notify_tip(supporter_name, amount, potions, message)


async def _handle_membership_started(data: dict):
    """Handle membership.started — new member joined."""
    supporter_name = data.get("supporter_name", "Anonymous")
    tier_name = data.get("membership_level_name", "Unknown Tier")
    amount = float(data.get("amount", 0))

    tier_id = TIER_MAP.get(tier_name, "unknown")
    logger.info(f"New member: {supporter_name} joined {tier_name} (${amount:.2f}/mo)")

    _append_supporter_event("membership_started", {
        "name": supporter_name,
        "tier": tier_name,
        "tier_id": tier_id,
        "amount": amount,
    })
    _update_stats("membership_started", amount)
    await notify_membership_started(supporter_name, tier_name, amount)


async def _handle_membership_cancelled(data: dict):
    """Handle membership.cancelled — member left."""
    supporter_name = data.get("supporter_name", "Anonymous")
    tier_name = data.get("membership_level_name", "Unknown Tier")

    logger.info(f"Membership cancelled: {supporter_name} left {tier_name}")

    _append_supporter_event("membership_cancelled", {
        "name": supporter_name,
        "tier": tier_name,
    })
    _update_stats("membership_cancelled")
    await notify_membership_cancelled(supporter_name, tier_name)


# --- Stats Endpoint ---

@app.get("/webhooks/bmc/stats")
async def get_stats():
    """Get BMC supporter statistics."""
    stats = _load_json(STATS_FILE)
    totals = stats.get("totals", {})

    # Calculate active members (started - cancelled)
    active_members = totals.get("memberships_started", 0) - totals.get("memberships_cancelled", 0)

    return {
        "tips": {
            "count": totals.get("tips_count", 0),
            "revenue": round(totals.get("tips_revenue", 0), 2),
        },
        "shop": {
            "count": totals.get("shop_count", 0),
            "revenue": round(totals.get("shop_revenue", 0), 2),
        },
        "memberships": {
            "active": max(0, active_members),
            "total_started": totals.get("memberships_started", 0),
            "total_cancelled": totals.get("memberships_cancelled", 0),
            "monthly_revenue": round(totals.get("membership_revenue", 0), 2),
        },
        "total_revenue": round(
            totals.get("tips_revenue", 0)
            + totals.get("shop_revenue", 0)
            + totals.get("membership_revenue", 0),
            2,
        ),
        "last_updated": stats.get("last_updated"),
    }


@app.get("/webhooks/bmc/recent")
async def get_recent_events(limit: int = 20):
    """Get recent supporter events."""
    log = _load_json(SUPPORTERS_LOG)
    events = log.get("events", [])
    return {"events": events[-limit:]}


# --- Health Check ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "bmc-webhook-handler",
        "port": WEBHOOK_PORT,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=WEBHOOK_PORT)
