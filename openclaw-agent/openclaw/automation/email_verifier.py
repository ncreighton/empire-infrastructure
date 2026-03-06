"""EmailVerifier — IMAP inbox monitor for auto-clicking verification links.

Connects to email inbox via IMAP, watches for verification emails from known platforms,
extracts verification URLs, and opens them via HTTP (or headless browser) to complete
verification. Designed for hands-free platform signup automation.
"""

from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ─── Known Platform Senders ──────────────────────────────────────────────────

PLATFORM_EMAIL_SENDERS: dict[str, list[str]] = {
    "gumroad": ["no-reply@gumroad.com", "noreply@gumroad.com", "support@gumroad.com"],
    "lemon_squeezy": ["no-reply@lemonsqueezy.com", "noreply@lemonsqueezy.com"],
    "etsy": ["no-reply@etsy.com", "noreply@etsy.com", "transaction@etsy.com"],
    "creative_market": ["no-reply@creativemarket.com"],
    "envato": ["no-reply@envato.com", "do-not-reply@envato.com"],
    "teachable": ["no-reply@teachable.com"],
    "thinkific": ["no-reply@thinkific.com"],
    "udemy": ["no-reply@udemy.com", "noreply@e.udemymail.com"],
    "promptbase": ["no-reply@promptbase.com"],
    "sendowl": ["no-reply@sendowl.com"],
    "make_marketplace": ["no-reply@make.com", "noreply@make.com"],
    "cgtrader": ["no-reply@cgtrader.com", "noreply@cgtrader.com"],
    "n8n_creator_hub": ["no-reply@n8n.io"],
    "payhip": ["no-reply@payhip.com"],
    "whop": ["no-reply@whop.com"],
    "skillshare": ["no-reply@skillshare.com"],
}

# ─── URL Extraction Patterns ────────────────────────────────────────────────

VERIFICATION_URL_PATTERNS: list[str] = [
    # href-enclosed URLs containing verification keywords
    r'href=["\']?(https?://[^\s"\'<>]+(?:verify|confirm|activate|validate|email)[^\s"\'<>]*)["\']?',
    # href-enclosed URLs with token/code/key params (common for verification)
    r'href=["\']?(https?://[^\s"\'<>]+(?:token|code|key)=[^\s"\'<>]*)["\']?',
    # Plain URLs containing verification keywords (fallback for plain-text emails)
    r'(https?://[^\s<>]+(?:verify|confirm|activate|validate)[^\s<>]*)',
]

# ─── Subject Line Detection ─────────────────────────────────────────────────

VERIFICATION_SUBJECT_PATTERNS: list[re.Pattern] = [
    re.compile(r'(?i)verify.*email'),
    re.compile(r'(?i)confirm.*email'),
    re.compile(r'(?i)email.*verification'),
    re.compile(r'(?i)email.*confirmation'),
    re.compile(r'(?i)activate.*account'),
    re.compile(r'(?i)complete.*registration'),
    re.compile(r'(?i)welcome.*confirm'),
    re.compile(r'(?i)action.*required.*verify'),
    re.compile(r'(?i)please.*verify'),
    re.compile(r'(?i)one.*more.*step'),
]

# Body keyword patterns (supplement subject line detection)
VERIFICATION_BODY_KEYWORDS: list[str] = [
    "verify your email",
    "confirm your email",
    "activate your account",
    "complete your registration",
    "click the link below to verify",
    "click the button below to confirm",
    "verify your account",
    "confirm your account",
]


@dataclass
class VerificationEmail:
    """Represents a detected verification email."""

    platform_id: str
    sender: str
    subject: str
    verification_url: str
    received_at: datetime
    message_id: str = ""
    verified: bool = False
    verified_at: datetime | None = None
    http_status: int = 0


@dataclass
class EmailVerifierConfig:
    """IMAP connection and polling configuration."""

    imap_host: str = ""
    imap_port: int = 993
    email_address: str = ""
    email_password: str = ""
    use_ssl: bool = True
    inbox_folder: str = "INBOX"
    check_interval_seconds: int = 15
    max_email_age_minutes: int = 30


class EmailVerifier:
    """Monitor email inbox for verification emails and auto-click verification links.

    Usage::

        verifier = EmailVerifier()
        if verifier.is_configured:
            verified = await verifier.auto_verify("gumroad", timeout_seconds=300)
    """

    def __init__(self, config: EmailVerifierConfig | None = None):
        self.config = config or self._load_config_from_env()
        self._connection: imaplib.IMAP4_SSL | imaplib.IMAP4 | None = None
        self.verified_emails: list[VerificationEmail] = []
        self.pending_platforms: set[str] = set()
        self._running = False
        self._processed_message_ids: set[str] = set()

    @staticmethod
    def _load_config_from_env() -> EmailVerifierConfig:
        """Load IMAP config from environment variables."""
        return EmailVerifierConfig(
            imap_host=os.environ.get("OPENCLAW_IMAP_HOST", "imap.gmail.com"),
            imap_port=int(os.environ.get("OPENCLAW_IMAP_PORT", "993")),
            email_address=os.environ.get("OPENCLAW_EMAIL", ""),
            email_password=os.environ.get("OPENCLAW_EMAIL_PASSWORD", ""),
        )

    @property
    def is_configured(self) -> bool:
        """Check if email credentials are provided."""
        return bool(
            self.config.imap_host
            and self.config.email_address
            and self.config.email_password
        )

    # ─── Connection Management ───────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to IMAP server and select inbox folder."""
        if not self.is_configured:
            logger.error("Email verifier not configured — missing IMAP credentials")
            return False

        try:
            if self.config.use_ssl:
                self._connection = imaplib.IMAP4_SSL(
                    self.config.imap_host, self.config.imap_port
                )
            else:
                self._connection = imaplib.IMAP4(
                    self.config.imap_host, self.config.imap_port
                )

            self._connection.login(
                self.config.email_address, self.config.email_password
            )
            self._connection.select(self.config.inbox_folder)
            logger.info(
                f"Connected to IMAP: {self.config.imap_host} as {self.config.email_address}"
            )
            return True

        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP login failed: {e}")
            self._connection = None
            return False
        except (OSError, TimeoutError) as e:
            logger.error(f"IMAP connection failed: {e}")
            self._connection = None
            return False

    def disconnect(self) -> None:
        """Close IMAP connection gracefully."""
        if self._connection:
            try:
                self._connection.close()
            except (imaplib.IMAP4.error, OSError) as e:
                logger.debug(f"IMAP close warning: {e}")
            try:
                self._connection.logout()
            except (imaplib.IMAP4.error, OSError) as e:
                logger.debug(f"IMAP logout warning: {e}")
            self._connection = None
            logger.info("IMAP connection closed")

    def _ensure_connected(self) -> bool:
        """Reconnect if the connection was dropped."""
        if self._connection is None:
            return self.connect()
        try:
            self._connection.noop()
            return True
        except Exception:
            logger.warning("IMAP connection lost — reconnecting")
            self._connection = None
            return self.connect()

    # ─── Blocking Watch ──────────────────────────────────────────────────

    def watch_for_verification(
        self, platform_id: str, timeout_seconds: int = 300
    ) -> VerificationEmail | None:
        """Watch inbox for a verification email from a specific platform.

        Blocks until a matching email is found or the timeout is reached.
        Polls at the interval configured in ``check_interval_seconds``.
        """
        if not self._ensure_connected():
            return None

        self.pending_platforms.add(platform_id)
        deadline = time.time() + timeout_seconds
        logger.info(
            f"Watching for {platform_id} verification email (timeout={timeout_seconds}s)"
        )

        while time.time() < deadline:
            emails = self.check_inbox(since_minutes=self.config.max_email_age_minutes)
            for ve in emails:
                if ve.platform_id == platform_id:
                    self.pending_platforms.discard(platform_id)
                    logger.info(f"Found verification email for {platform_id}")
                    return ve

            remaining = deadline - time.time()
            sleep_time = min(self.config.check_interval_seconds, max(remaining, 0))
            if sleep_time <= 0:
                break
            time.sleep(sleep_time)

        self.pending_platforms.discard(platform_id)
        logger.warning(f"Timeout waiting for {platform_id} verification email")
        return None

    # ─── Async Watch ─────────────────────────────────────────────────────

    async def watch_for_verification_async(
        self, platform_id: str, timeout_seconds: int = 300
    ) -> VerificationEmail | None:
        """Async version of watch_for_verification — polls inbox periodically."""
        if not self._ensure_connected():
            return None

        self.pending_platforms.add(platform_id)
        deadline = time.time() + timeout_seconds
        logger.info(
            f"Async watching for {platform_id} verification email (timeout={timeout_seconds}s)"
        )

        while time.time() < deadline:
            # Run IMAP check in thread to avoid blocking the event loop
            emails = await asyncio.get_event_loop().run_in_executor(
                None,
                self.check_inbox,
                self.config.max_email_age_minutes,
            )
            for ve in emails:
                if ve.platform_id == platform_id:
                    self.pending_platforms.discard(platform_id)
                    logger.info(f"Found verification email for {platform_id}")
                    return ve

            remaining = deadline - time.time()
            sleep_time = min(self.config.check_interval_seconds, max(remaining, 0))
            if sleep_time <= 0:
                break
            await asyncio.sleep(sleep_time)

        self.pending_platforms.discard(platform_id)
        logger.warning(f"Timeout waiting for {platform_id} verification email")
        return None

    # ─── Inbox Scanning ──────────────────────────────────────────────────

    def check_inbox(self, since_minutes: int = 30) -> list[VerificationEmail]:
        """Check inbox for any verification emails from known platforms.

        Returns newly found verification emails (skips already-processed message IDs).
        """
        if not self._ensure_connected():
            return []

        results: list[VerificationEmail] = []
        raw_emails = self._search_emails(since_minutes)

        for msg_id, raw_data in raw_emails:
            if msg_id in self._processed_message_ids:
                continue

            parsed = self._parse_email(raw_data)
            if not parsed:
                continue

            sender = parsed["sender"]
            subject = parsed["subject"]
            body = parsed["body"]

            platform_id = self._detect_platform(sender)
            if platform_id is None:
                continue

            if not self._is_verification_email(subject, body):
                continue

            url = self._extract_verification_url(body)
            if not url:
                logger.warning(
                    f"Verification email detected for {platform_id} but no URL found"
                )
                continue

            ve = VerificationEmail(
                platform_id=platform_id,
                sender=sender,
                subject=subject,
                verification_url=url,
                received_at=parsed.get("date", datetime.now()),
                message_id=msg_id,
            )
            results.append(ve)
            self._processed_message_ids.add(msg_id)
            logger.info(
                f"Verification email found: platform={platform_id}, "
                f"subject={subject!r}, url={url[:80]}..."
            )

        return results

    def _search_emails(self, since_minutes: int = 30) -> list[tuple[str, bytes]]:
        """Search IMAP for recent unseen emails.

        Returns a list of (message_id, raw_bytes) tuples.
        """
        if not self._connection:
            return []

        try:
            # Build IMAP search criteria: UNSEEN emails since a date
            since_date = datetime.now() - timedelta(minutes=since_minutes)
            date_str = since_date.strftime("%d-%b-%Y")
            search_criteria = f'(UNSEEN SINCE {date_str})'

            status, data = self._connection.search(None, search_criteria)
            if status != "OK" or not data or not data[0]:
                return []

            message_nums = data[0].split()
            results: list[tuple[str, bytes]] = []

            for num in message_nums:
                try:
                    status, msg_data = self._connection.fetch(num, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        continue
                    raw_bytes = msg_data[0][1]
                    if isinstance(raw_bytes, bytes):
                        msg_id = num.decode() if isinstance(num, bytes) else str(num)
                        results.append((msg_id, raw_bytes))
                except Exception as e:
                    logger.debug(f"Failed to fetch email {num}: {e}")
                    continue

            return results

        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during email search: {e}")
            return []

    def _parse_email(self, raw_email: bytes) -> dict[str, Any] | None:
        """Parse raw email bytes into sender, subject, body, and date.

        Handles multipart MIME messages, extracting both plain text and HTML bodies.
        """
        try:
            msg = email.message_from_bytes(raw_email)

            # Decode subject
            subject = ""
            raw_subject = msg.get("Subject", "")
            if raw_subject:
                decoded_parts = decode_header(raw_subject)
                subject_parts = []
                for part, charset in decoded_parts:
                    if isinstance(part, bytes):
                        subject_parts.append(
                            part.decode(charset or "utf-8", errors="replace")
                        )
                    else:
                        subject_parts.append(part)
                subject = " ".join(subject_parts)

            # Decode sender
            sender = ""
            raw_from = msg.get("From", "")
            if raw_from:
                # Extract just the email address from "Name <email>" format
                match = re.search(r'<([^>]+)>', raw_from)
                sender = match.group(1).lower() if match else raw_from.lower().strip()

            # Parse date
            received_date = datetime.now()
            raw_date = msg.get("Date")
            if raw_date:
                try:
                    received_date = parsedate_to_datetime(raw_date)
                except (ValueError, TypeError):
                    pass

            # Extract body (prefer HTML for link extraction, fallback to plain text)
            body = ""
            html_body = ""
            plain_body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))

                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue

                    try:
                        payload = part.get_payload(decode=True)
                        if payload is None:
                            continue
                        charset = part.get_content_charset() or "utf-8"
                        decoded = payload.decode(charset, errors="replace")
                    except Exception as decode_err:
                        logger.debug(f"MIME part decode failed: {decode_err}")
                        continue

                    if content_type == "text/html":
                        html_body = decoded
                    elif content_type == "text/plain":
                        plain_body = decoded
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or "utf-8"
                        decoded = payload.decode(charset, errors="replace")
                        if msg.get_content_type() == "text/html":
                            html_body = decoded
                        else:
                            plain_body = decoded
                except Exception as body_err:
                    logger.debug(f"Email body decode failed: {body_err}")

            # Prefer HTML body (has href links), fall back to plain text
            body = html_body or plain_body

            if not body:
                return None

            return {
                "sender": sender,
                "subject": subject,
                "body": body,
                "date": received_date,
            }

        except Exception as e:
            logger.error(f"Failed to parse email: {e}")
            return None

    # ─── Detection Logic ─────────────────────────────────────────────────

    def _detect_platform(self, sender: str) -> str | None:
        """Match a sender email address to a platform_id.

        Checks exact matches first, then falls back to domain matching.
        """
        sender_lower = sender.lower().strip()

        for platform_id, addresses in PLATFORM_EMAIL_SENDERS.items():
            for addr in addresses:
                if sender_lower == addr.lower():
                    return platform_id

        # Fallback: extract domain and try partial match
        match = re.search(r'@([a-zA-Z0-9.-]+)$', sender_lower)
        if match:
            sender_domain = match.group(1)
            for platform_id, addresses in PLATFORM_EMAIL_SENDERS.items():
                for addr in addresses:
                    addr_match = re.search(r'@([a-zA-Z0-9.-]+)$', addr)
                    if addr_match and addr_match.group(1) == sender_domain:
                        return platform_id

        return None

    def _is_verification_email(self, subject: str, body: str) -> bool:
        """Check if an email is a verification email based on subject and body patterns."""
        # Check subject
        for pattern in VERIFICATION_SUBJECT_PATTERNS:
            if pattern.search(subject):
                return True

        # Check body keywords
        body_lower = body.lower()
        for keyword in VERIFICATION_BODY_KEYWORDS:
            if keyword in body_lower:
                return True

        return False

    def _extract_verification_url(self, body: str) -> str | None:
        """Extract the verification URL from an email body.

        Tries href-based patterns first (HTML), then plain-text URL patterns.
        Returns the first match that looks like a verification link.
        """
        for pattern in VERIFICATION_URL_PATTERNS:
            matches = re.findall(pattern, body, re.IGNORECASE)
            for url in matches:
                # Clean up the URL (remove trailing quotes, angle brackets, etc.)
                url = url.rstrip('"\'>;) ')
                # Decode HTML entities
                url = url.replace("&amp;", "&")
                # Skip obviously wrong URLs (CSS, images, unsubscribe links)
                lower_url = url.lower()
                if any(skip in lower_url for skip in [
                    "unsubscribe", ".css", ".png", ".jpg", ".gif", ".ico",
                    "privacy", "terms", "preferences", "logo",
                ]):
                    continue
                return url

        return None

    # ─── Verification Clicks ─────────────────────────────────────────────

    async def click_verification_link(self, url: str) -> bool:
        """Open verification URL via HTTP GET (works for most platforms).

        Sends a request with realistic browser headers and follows redirects.
        Returns True if the final response was a 2xx status.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                headers=headers,
            ) as client:
                response = await client.get(url)
                status = response.status_code
                logger.info(
                    f"Verification link response: {status} "
                    f"(final URL: {response.url})"
                )

                if 200 <= status < 400:
                    # Check response body for success indicators
                    body_lower = response.text.lower()
                    if any(
                        indicator in body_lower
                        for indicator in [
                            "verified", "confirmed", "success",
                            "thank you", "email has been verified",
                            "account activated", "welcome",
                        ]
                    ):
                        logger.info("Verification confirmed by response content")
                    return True

                logger.warning(f"Verification link returned status {status}")
                return False

        except httpx.TimeoutException:
            logger.error(f"Timeout clicking verification link: {url[:80]}...")
            return False
        except httpx.HTTPError as e:
            logger.error(f"HTTP error clicking verification link: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error clicking verification link: {e}")
            return False

    async def click_verification_link_browser(self, url: str) -> bool:
        """Open verification URL in a headless browser (for JS-required verifications).

        Falls back to this when the simple HTTP GET approach does not work because
        the platform requires JavaScript execution to complete verification.
        """
        try:
            from openclaw.browser.browser_manager import BrowserManager

            async with BrowserManager(headless=True) as browser:
                result = await browser.run_agent(
                    task=(
                        f"Navigate to this verification URL and wait for the page to "
                        f"load completely. If there is a button to confirm or verify, "
                        f"click it. URL: {url}"
                    ),
                    max_steps=5,
                )
                success = result.get("success", False)
                if success:
                    logger.info("Verification completed via headless browser")
                else:
                    logger.warning(
                        f"Browser verification may have failed: {result.get('error', 'unknown')}"
                    )
                return success

        except ImportError:
            logger.error(
                "browser-use not installed — cannot use browser verification fallback"
            )
            return False
        except Exception as e:
            logger.error(f"Browser verification failed: {e}")
            return False

    # ─── Full Auto-Verify ────────────────────────────────────────────────

    async def auto_verify(
        self, platform_id: str, timeout_seconds: int = 300
    ) -> bool:
        """Full auto-verification: watch inbox, extract URL, click link, confirm.

        This is the primary high-level method. It:
        1. Watches the inbox for a verification email from the given platform
        2. Extracts the verification URL from the email body
        3. Clicks the link via HTTP (falling back to headless browser)
        4. Records the result

        Returns True if verification was completed successfully.
        """
        logger.info(f"Starting auto-verify for {platform_id}")

        ve = await self.watch_for_verification_async(
            platform_id, timeout_seconds=timeout_seconds
        )
        if ve is None:
            logger.error(f"No verification email found for {platform_id}")
            return False

        logger.info(f"Attempting to click verification link for {platform_id}")

        # Try simple HTTP first
        success = await self.click_verification_link(ve.verification_url)

        # If HTTP approach fails, try browser fallback
        if not success:
            logger.info(
                f"HTTP verification failed for {platform_id}, trying browser fallback"
            )
            success = await self.click_verification_link_browser(ve.verification_url)

        ve.verified = success
        ve.verified_at = datetime.now() if success else None
        self.verified_emails.append(ve)

        if success:
            logger.info(f"Auto-verification successful for {platform_id}")
        else:
            logger.error(f"Auto-verification failed for {platform_id}")

        return success

    # ─── Queries ─────────────────────────────────────────────────────────

    def get_verified_platforms(self) -> list[str]:
        """Get list of platform IDs that have been successfully verified."""
        return [e.platform_id for e in self.verified_emails if e.verified]

    def get_stats(self) -> dict[str, Any]:
        """Get verification statistics."""
        total = len(self.verified_emails)
        verified = sum(1 for e in self.verified_emails if e.verified)
        failed = total - verified

        return {
            "total_processed": total,
            "verified": verified,
            "failed": failed,
            "pending_platforms": list(self.pending_platforms),
            "verified_platforms": self.get_verified_platforms(),
            "processed_message_ids": len(self._processed_message_ids),
            "is_connected": self._connection is not None,
            "is_configured": self.is_configured,
        }

    def add_platform_sender(self, platform_id: str, sender_email: str) -> None:
        """Register a new sender email for a platform (runtime-only)."""
        if platform_id not in PLATFORM_EMAIL_SENDERS:
            PLATFORM_EMAIL_SENDERS[platform_id] = []
        if sender_email.lower() not in PLATFORM_EMAIL_SENDERS[platform_id]:
            PLATFORM_EMAIL_SENDERS[platform_id].append(sender_email.lower())
            logger.info(
                f"Added sender {sender_email} for platform {platform_id}"
            )
