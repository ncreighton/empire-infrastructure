"""Email inbox health check — scan for pending verification emails.

SCAN tier: runs every 30 minutes.
"""

from __future__ import annotations

import logging
import os
import time

from openclaw.models import CheckResult, HealthCheck, HeartbeatTier

logger = logging.getLogger(__name__)


async def check_inbox() -> list[HealthCheck]:
    """Check IMAP inbox for pending verification emails.

    Returns:
        List with one HealthCheck for email inbox status.
    """
    imap_host = os.environ.get("OPENCLAW_IMAP_HOST", "")
    email_addr = os.environ.get("OPENCLAW_EMAIL", "")
    email_pass = os.environ.get("OPENCLAW_EMAIL_PASSWORD", "")

    if not all([imap_host, email_addr, email_pass]):
        return [HealthCheck(
            name="email:inbox",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.UNKNOWN,
            message="IMAP not configured (OPENCLAW_IMAP_HOST / OPENCLAW_EMAIL_PASSWORD)",
        )]

    start = time.monotonic()
    try:
        import imaplib

        imap_port = int(os.environ.get("OPENCLAW_IMAP_PORT", "993"))
        mail = imaplib.IMAP4_SSL(imap_host, imap_port)
        mail.login(email_addr, email_pass)
        mail.select("INBOX")

        # Search for verification-related emails
        _, data = mail.search(None, '(UNSEEN SUBJECT "verify")')
        verify_ids = data[0].split() if data[0] else []

        _, data2 = mail.search(None, '(UNSEEN SUBJECT "confirm")')
        confirm_ids = data2[0].split() if data2[0] else []

        # Deduplicate
        pending_ids = set(verify_ids) | set(confirm_ids)
        pending_count = len(pending_ids)

        mail.logout()

        duration_ms = (time.monotonic() - start) * 1000

        if pending_count > 0:
            return [HealthCheck(
                name="email:inbox",
                tier=HeartbeatTier.SCAN,
                result=CheckResult.DEGRADED,
                message=f"{pending_count} pending verification email(s)",
                details={"pending_count": pending_count},
                duration_ms=duration_ms,
            )]
        else:
            return [HealthCheck(
                name="email:inbox",
                tier=HeartbeatTier.SCAN,
                result=CheckResult.HEALTHY,
                message="No pending verification emails",
                duration_ms=duration_ms,
            )]

    except Exception as e:
        return [HealthCheck(
            name="email:inbox",
            tier=HeartbeatTier.SCAN,
            result=CheckResult.DOWN,
            message=f"IMAP check failed: {str(e)[:80]}",
            duration_ms=(time.monotonic() - start) * 1000,
        )]
