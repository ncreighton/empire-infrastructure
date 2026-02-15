"""
Email Agent — OpenClaw Empire Email Account Management

Automates email account creation (Gmail, Yahoo, Outlook/Hotmail), login,
inbox reading via OCR, compose/send/reply/forward, verification code/link
extraction, folder/label management, attachment handling, and multi-account
monitoring. Credentials stored in AccountManager, not locally.

Data persisted to: data/email/

Usage:
    from src.email_agent import EmailAgent, get_email_agent

    agent = get_email_agent()
    await agent.create_gmail_account("john.doe", "SecurePass123!")
    inbox = await agent.read_inbox("john.doe@gmail.com")
    code = await agent.wait_for_verification("john.doe@gmail.com", sender="noreply@instagram.com")

CLI:
    python -m src.email_agent create --provider gmail --username johndoe
    python -m src.email_agent inbox --account john.doe@gmail.com --limit 10
    python -m src.email_agent send --from john@gmail.com --to test@example.com --subject "Hello"
    python -m src.email_agent verify --account john@gmail.com --sender noreply@instagram.com
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("email_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "email"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class EmailProvider(str, Enum):
    GMAIL = "gmail"
    YAHOO = "yahoo"
    OUTLOOK = "outlook"
    HOTMAIL = "hotmail"
    PROTONMAIL = "protonmail"
    ICLOUD = "icloud"
    OTHER = "other"


class EmailStatus(str, Enum):
    UNREAD = "unread"
    READ = "read"
    STARRED = "starred"
    ARCHIVED = "archived"
    DELETED = "deleted"
    SPAM = "spam"
    DRAFT = "draft"


class AccountStatus(str, Enum):
    ACTIVE = "active"
    LOCKED = "locked"
    SUSPENDED = "suspended"
    NEEDS_VERIFICATION = "needs_verification"
    UNKNOWN = "unknown"


class VerificationType(str, Enum):
    CODE = "code"        # 6-digit numeric code
    LINK = "link"        # Clickable verification link
    BUTTON = "button"    # "Verify" button in email
    PIN = "pin"          # 4-digit PIN
    UNKNOWN = "unknown"


PROVIDER_PACKAGES = {
    EmailProvider.GMAIL: "com.google.android.gm",
    EmailProvider.YAHOO: "com.yahoo.mobile.client.android.mail",
    EmailProvider.OUTLOOK: "com.microsoft.office.outlook",
}

PROVIDER_WEB_URLS = {
    EmailProvider.GMAIL: "https://mail.google.com",
    EmailProvider.YAHOO: "https://mail.yahoo.com",
    EmailProvider.OUTLOOK: "https://outlook.live.com",
    EmailProvider.HOTMAIL: "https://outlook.live.com",
    EmailProvider.PROTONMAIL: "https://mail.proton.me",
}

SIGNUP_URLS = {
    EmailProvider.GMAIL: "https://accounts.google.com/signup",
    EmailProvider.YAHOO: "https://login.yahoo.com/account/create",
    EmailProvider.OUTLOOK: "https://signup.live.com",
    EmailProvider.HOTMAIL: "https://signup.live.com",
}


@dataclass
class EmailAccount:
    """An email account tracked by the agent."""
    email: str = ""
    provider: EmailProvider = EmailProvider.OTHER
    username: str = ""
    status: AccountStatus = AccountStatus.ACTIVE
    created_at: str = field(default_factory=_now_iso)
    last_checked: str = ""
    last_login: str = ""
    unread_count: int = 0
    total_emails: int = 0
    recovery_email: str = ""
    recovery_phone: str = ""
    credential_id: str = ""  # Reference to AccountManager
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["provider"] = self.provider.value
        d["status"] = self.status.value
        return d


@dataclass
class EmailMessage:
    """A parsed email message from OCR extraction."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    account: str = ""
    sender: str = ""
    subject: str = ""
    preview: str = ""
    body: str = ""
    timestamp: str = ""
    status: EmailStatus = EmailStatus.UNREAD
    has_attachment: bool = False
    labels: List[str] = field(default_factory=list)
    reply_to: str = ""
    cc: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class VerificationResult:
    """Result of a verification code/link extraction."""
    found: bool = False
    verification_type: VerificationType = VerificationType.UNKNOWN
    code: str = ""
    link: str = ""
    sender: str = ""
    subject: str = ""
    extracted_at: str = field(default_factory=_now_iso)
    raw_text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verification_type"] = self.verification_type.value
        return d


@dataclass
class ComposeEmail:
    """Email composition data."""
    to: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    attachments: List[str] = field(default_factory=list)
    reply_to_id: str = ""
    forward_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# EmailAgent
# ---------------------------------------------------------------------------

class EmailAgent:
    """
    Email account management agent for Android phones.

    Creates email accounts, reads inboxes via OCR, sends emails,
    and extracts verification codes/links. Uses BrowserController
    for web-based flows and PhoneController for app-based flows.

    Usage:
        agent = get_email_agent()
        await agent.create_gmail_account("john.doe", "SecurePass123!")
        inbox = await agent.read_inbox("john.doe@gmail.com")
    """

    def __init__(
        self,
        controller: Any = None,
        browser: Any = None,
        account_mgr: Any = None,
        data_dir: Optional[Path] = None,
    ):
        self._controller = controller
        self._browser = browser
        self._account_mgr = account_mgr
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._accounts: Dict[str, EmailAccount] = {}
        self._messages: Dict[str, List[EmailMessage]] = {}  # account -> messages
        self._verification_log: List[VerificationResult] = []

        self._load_state()
        logger.info("EmailAgent initialized (%d accounts)", len(self._accounts))

    # ── Property helpers ──

    @property
    def controller(self):
        if self._controller is None:
            try:
                from src.phone_controller import PhoneController
                self._controller = PhoneController()
            except ImportError:
                logger.error("PhoneController not available")
        return self._controller

    @property
    def browser(self):
        if self._browser is None:
            try:
                from src.browser_controller import get_browser
                self._browser = get_browser()
            except ImportError:
                logger.warning("BrowserController not available")
        return self._browser

    @property
    def account_mgr(self):
        if self._account_mgr is None:
            try:
                from src.account_manager import get_account_manager
                self._account_mgr = get_account_manager()
            except ImportError:
                logger.warning("AccountManager not available")
        return self._account_mgr

    # ── Persistence ──

    def _load_state(self) -> None:
        state = _load_json(self._data_dir / "state.json")
        for email, data in state.get("accounts", {}).items():
            if isinstance(data, dict):
                self._accounts[email] = EmailAccount(**data)
        for email, msgs in state.get("messages", {}).items():
            self._messages[email] = [
                EmailMessage(**m) if isinstance(m, dict) else m
                for m in msgs[-50:]  # Keep last 50 per account
            ]

    def _save_state(self) -> None:
        _save_json(self._data_dir / "state.json", {
            "accounts": {k: v.to_dict() for k, v in self._accounts.items()},
            "messages": {
                k: [m.to_dict() for m in v[-50:]]
                for k, v in self._messages.items()
            },
            "updated_at": _now_iso(),
        })

    # ── ADB / vision helpers ──

    async def _adb_shell(self, cmd: str) -> str:
        if self.controller is None:
            raise RuntimeError("PhoneController not available")
        return await self.controller._adb_shell(cmd)

    async def _take_screenshot(self) -> str:
        return await self.controller.screenshot()

    async def _analyze_screen(self, screenshot_path: str) -> dict:
        if hasattr(self.controller, '_vision') and self.controller._vision:
            result = await self.controller._vision.analyze_screen(screenshot_path=screenshot_path)
            return result if isinstance(result, dict) else {"raw": str(result)}
        # Fallback: try importing VisionAgent
        try:
            from src.vision_agent import VisionAgent
            va = VisionAgent()
            result = await va.analyze_screen(screenshot_path=screenshot_path)
            return result if isinstance(result, dict) else {"raw": str(result)}
        except ImportError:
            return {}

    async def _find_element(self, description: str, screenshot_path: str = None) -> Optional[dict]:
        try:
            from src.vision_agent import VisionAgent
            va = VisionAgent()
            kwargs = {"description": description}
            if screenshot_path:
                kwargs["screenshot_path"] = screenshot_path
            result = await va.find_element(**kwargs)
            if isinstance(result, dict) and result.get("x") is not None:
                return result
        except ImportError:
            pass
        return None

    # ── Account creation ──

    async def create_gmail_account(
        self,
        username: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
        recovery_email: str = "",
        birth_year: int = 1990,
        birth_month: int = 6,
        birth_day: int = 15,
    ) -> Dict[str, Any]:
        """
        Create a Gmail account via the browser.

        Args:
            username: Desired username (without @gmail.com).
            password: Account password.
            first_name: First name for the account.
            last_name: Last name for the account.
            recovery_email: Optional recovery email.
            birth_year/month/day: Date of birth.

        Returns:
            Dict with success status and account info.
        """
        email = f"{username}@gmail.com"

        if not self.browser:
            return {"success": False, "error": "BrowserController not available"}

        try:
            # Navigate to Google signup
            await self.browser.open_url(SIGNUP_URLS[EmailProvider.GMAIL], wait_for_load=True)
            await asyncio.sleep(2.0)

            # Fill first name
            if first_name:
                await self.browser.fill_form({"First name": first_name})
                await asyncio.sleep(0.5)

            # Fill last name
            if last_name:
                await self.browser.fill_form({"Last name": last_name})
                await asyncio.sleep(0.5)

            # Click Next
            await self.browser.click_element("Next button")
            await asyncio.sleep(2.0)

            # Fill birthday and gender
            await self.browser.fill_form({
                "Month": str(birth_month),
                "Day": str(birth_day),
                "Year": str(birth_year),
            })
            await asyncio.sleep(0.5)

            await self.browser.click_element("Next button")
            await asyncio.sleep(2.0)

            # Choose username
            await self.browser.fill_form({"Username": username})
            await asyncio.sleep(0.5)

            await self.browser.click_element("Next button")
            await asyncio.sleep(2.0)

            # Fill password
            await self.browser.fill_form({
                "Password": password,
                "Confirm": password,
            })
            await asyncio.sleep(0.5)

            await self.browser.click_element("Next button")
            await asyncio.sleep(2.0)

            # Skip phone number if possible
            screenshot = await self._take_screenshot()
            skip_btn = await self._find_element("Skip button", screenshot)
            if skip_btn:
                await self.controller.tap(skip_btn["x"], skip_btn["y"])
                await asyncio.sleep(1.0)

            # Accept terms
            await self.browser.click_element("I agree button or Agree button")
            await asyncio.sleep(3.0)

            # Store credentials in AccountManager
            cred_id = ""
            if self.account_mgr:
                cred_id = self.account_mgr.store_credential_sync(
                    platform="gmail",
                    username=email,
                    password=password,
                    metadata={"first_name": first_name, "last_name": last_name},
                ) if hasattr(self.account_mgr, 'store_credential_sync') else ""

            # Register account
            account = EmailAccount(
                email=email,
                provider=EmailProvider.GMAIL,
                username=username,
                status=AccountStatus.ACTIVE,
                recovery_email=recovery_email,
                credential_id=cred_id,
            )
            self._accounts[email] = account
            self._save_state()

            logger.info("Created Gmail account: %s", email)
            return {"success": True, "email": email, "account": account.to_dict()}

        except Exception as exc:
            logger.error("Gmail creation failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def create_outlook_account(
        self,
        username: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
    ) -> Dict[str, Any]:
        """Create an Outlook/Hotmail account via the browser."""
        email = f"{username}@outlook.com"

        if not self.browser:
            return {"success": False, "error": "BrowserController not available"}

        try:
            await self.browser.open_url(SIGNUP_URLS[EmailProvider.OUTLOOK], wait_for_load=True)
            await asyncio.sleep(2.0)

            # Fill email
            await self.browser.fill_form({"New email": username})
            await asyncio.sleep(0.5)

            await self.browser.click_element("Next button")
            await asyncio.sleep(2.0)

            # Fill password
            await self.browser.fill_form({"Create password": password})
            await asyncio.sleep(0.5)

            await self.browser.click_element("Next button")
            await asyncio.sleep(2.0)

            # Fill name
            if first_name:
                await self.browser.fill_form({
                    "First name": first_name,
                    "Last name": last_name or "",
                })
                await asyncio.sleep(0.5)
                await self.browser.click_element("Next button")
                await asyncio.sleep(2.0)

            # Store credentials
            cred_id = ""
            if self.account_mgr:
                try:
                    result = self.account_mgr.store_credential(
                        platform="outlook",
                        username=email,
                        password=password,
                    )
                    cred_id = result if isinstance(result, str) else ""
                except Exception:
                    pass

            account = EmailAccount(
                email=email,
                provider=EmailProvider.OUTLOOK,
                username=username,
                status=AccountStatus.ACTIVE,
                credential_id=cred_id,
            )
            self._accounts[email] = account
            self._save_state()

            logger.info("Created Outlook account: %s", email)
            return {"success": True, "email": email, "account": account.to_dict()}

        except Exception as exc:
            logger.error("Outlook creation failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def create_yahoo_account(
        self,
        username: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
    ) -> Dict[str, Any]:
        """Create a Yahoo Mail account via the browser."""
        email = f"{username}@yahoo.com"

        if not self.browser:
            return {"success": False, "error": "BrowserController not available"}

        try:
            await self.browser.open_url(SIGNUP_URLS[EmailProvider.YAHOO], wait_for_load=True)
            await asyncio.sleep(2.0)

            fields = {"First name": first_name, "Last name": last_name, "Email address": username}
            await self.browser.fill_form(fields)
            await asyncio.sleep(0.5)

            await self.browser.fill_form({"Password": password})
            await asyncio.sleep(0.5)

            await self.browser.click_element("Continue button or Next button")
            await asyncio.sleep(3.0)

            # Store credentials
            if self.account_mgr:
                try:
                    self.account_mgr.store_credential(
                        platform="yahoo", username=email, password=password,
                    )
                except Exception:
                    pass

            account = EmailAccount(
                email=email,
                provider=EmailProvider.YAHOO,
                username=username,
                status=AccountStatus.ACTIVE,
            )
            self._accounts[email] = account
            self._save_state()

            logger.info("Created Yahoo account: %s", email)
            return {"success": True, "email": email, "account": account.to_dict()}

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Login ──

    async def login(self, email: str) -> Dict[str, Any]:
        """Log in to an email account via browser or app."""
        account = self._accounts.get(email)
        if not account:
            return {"success": False, "error": f"Account {email} not registered"}

        # Get credentials
        password = ""
        if self.account_mgr and account.credential_id:
            try:
                cred = self.account_mgr.get_credential(account.credential_id)
                if cred:
                    password = cred.get("password", "")
            except Exception:
                pass

        if not password:
            return {"success": False, "error": "No password found for account"}

        provider = account.provider
        login_url = PROVIDER_WEB_URLS.get(provider, "")

        if not login_url or not self.browser:
            return {"success": False, "error": "Cannot log in (no browser or URL)"}

        try:
            result = await self.browser.login(
                url=login_url,
                username=email,
                password=password,
                username_field="Email or phone" if provider == EmailProvider.GMAIL else "Email",
                password_field="Password",
                submit_label="Next" if provider == EmailProvider.GMAIL else "Sign in",
            )

            if result.get("success"):
                account.last_login = _now_iso()
                account.status = AccountStatus.ACTIVE
                self._save_state()

            return result

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Inbox reading ──

    async def read_inbox(
        self, email: str, limit: int = 10, use_app: bool = True
    ) -> List[EmailMessage]:
        """
        Read the inbox of an email account via OCR.

        Opens the email app or web interface, scrolls through messages,
        and extracts sender, subject, and preview via vision analysis.

        Args:
            email: Email address to check.
            limit: Maximum messages to extract.
            use_app: Whether to use the native app (vs browser).

        Returns:
            List of EmailMessage objects.
        """
        account = self._accounts.get(email)
        if not account:
            logger.warning("Account %s not registered", email)
            return []

        messages: List[EmailMessage] = []

        try:
            if use_app:
                package = PROVIDER_PACKAGES.get(account.provider)
                if package:
                    await self.controller.launch_app(package)
                    await asyncio.sleep(3.0)
                else:
                    # Fallback to browser
                    url = PROVIDER_WEB_URLS.get(account.provider, "")
                    if url and self.browser:
                        await self.browser.open_url(url, wait_for_load=True)
                        await asyncio.sleep(3.0)
            else:
                url = PROVIDER_WEB_URLS.get(account.provider, "")
                if url and self.browser:
                    await self.browser.open_url(url, wait_for_load=True)
                    await asyncio.sleep(3.0)

            # Extract messages by scrolling and reading
            for scroll_num in range(5):
                screenshot = await self._take_screenshot()
                analysis = await self._analyze_screen(screenshot)

                new_messages = self._parse_inbox_messages(analysis, email)
                messages.extend(new_messages)

                if len(messages) >= limit:
                    break

                await self.controller.scroll_down(600)
                await asyncio.sleep(1.5)

        except Exception as exc:
            logger.error("Inbox read failed for %s: %s", email, exc)

        messages = messages[:limit]

        # Update account
        if account:
            account.last_checked = _now_iso()
            account.unread_count = sum(1 for m in messages if m.status == EmailStatus.UNREAD)
            account.total_emails = len(messages)

        # Cache messages
        self._messages[email] = messages
        self._save_state()

        logger.info("Read %d messages from %s", len(messages), email)
        return messages

    def _parse_inbox_messages(self, analysis: dict, account: str) -> List[EmailMessage]:
        """Parse inbox messages from vision analysis."""
        messages = []
        if not isinstance(analysis, dict):
            return messages

        visible = analysis.get("visible_text", "")
        if isinstance(visible, list):
            visible = "\n".join(visible)

        # Simple heuristic: look for sender-subject pairs
        lines = [l.strip() for l in visible.split("\n") if l.strip()]

        i = 0
        while i < len(lines) - 1:
            # Look for lines that could be sender names followed by subjects
            sender_line = lines[i]
            subject_line = lines[i + 1] if i + 1 < len(lines) else ""

            # Heuristic: if the line contains @ or looks like a name/company
            if (len(sender_line) < 60 and len(subject_line) < 120 and
                not sender_line.startswith("http") and sender_line):
                msg = EmailMessage(
                    account=account,
                    sender=sender_line,
                    subject=subject_line,
                    preview=lines[i + 2] if i + 2 < len(lines) else "",
                )
                messages.append(msg)
                i += 3  # Skip sender + subject + preview
            else:
                i += 1

        return messages

    async def open_email(self, email: str, subject_or_sender: str) -> Dict[str, Any]:
        """Open a specific email by tapping on it."""
        try:
            screenshot = await self._take_screenshot()
            element = await self._find_element(
                f"email from '{subject_or_sender}' or with subject '{subject_or_sender}'",
                screenshot
            )
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(2.0)

                # Read the full email body
                body_screenshot = await self._take_screenshot()
                body_analysis = await self._analyze_screen(body_screenshot)
                body_text = ""
                if isinstance(body_analysis, dict):
                    visible = body_analysis.get("visible_text", "")
                    body_text = "\n".join(visible) if isinstance(visible, list) else str(visible)

                return {"success": True, "body": body_text}
            return {"success": False, "error": "Email not found on screen"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Compose / Send ──

    async def compose_and_send(
        self,
        from_account: str,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        attachment: str = "",
    ) -> Dict[str, Any]:
        """
        Compose and send an email.

        Args:
            from_account: Sender email address.
            to: Recipient email address.
            subject: Email subject.
            body: Email body text.
            cc: CC recipient (optional).
            attachment: Path to attachment file (optional).

        Returns:
            Dict with send result.
        """
        account = self._accounts.get(from_account)
        if not account:
            return {"success": False, "error": f"Account {from_account} not registered"}

        try:
            # Use Android intent for email composition
            cmd = (
                f"am start -a android.intent.action.SEND "
                f"-t 'text/plain' "
                f"--es android.intent.extra.EMAIL '{to}' "
                f"--es android.intent.extra.SUBJECT '{subject}' "
                f"--es android.intent.extra.TEXT '{body}'"
            )
            if cc:
                cmd += f" --es android.intent.extra.CC '{cc}'"

            await self._adb_shell(cmd)
            await asyncio.sleep(3.0)

            # Find and click Send button
            screenshot = await self._take_screenshot()
            send_btn = await self._find_element("Send button or send icon", screenshot)
            if send_btn:
                await self.controller.tap(send_btn["x"], send_btn["y"])
                await asyncio.sleep(2.0)
                logger.info("Email sent from %s to %s", from_account, to)
                return {"success": True, "from": from_account, "to": to, "subject": subject}
            else:
                return {"success": False, "error": "Send button not found"}

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def reply(self, email: str, body: str) -> Dict[str, Any]:
        """Reply to the currently open email."""
        try:
            # Find reply button
            screenshot = await self._take_screenshot()
            reply_btn = await self._find_element("Reply button or reply icon", screenshot)
            if not reply_btn:
                return {"success": False, "error": "Reply button not found"}

            await self.controller.tap(reply_btn["x"], reply_btn["y"])
            await asyncio.sleep(1.5)

            # Type reply body
            await self.controller.type_text(body)
            await asyncio.sleep(0.5)

            # Send
            screenshot2 = await self._take_screenshot()
            send_btn = await self._find_element("Send button", screenshot2)
            if send_btn:
                await self.controller.tap(send_btn["x"], send_btn["y"])
                await asyncio.sleep(2.0)
                return {"success": True, "action": "replied"}

            return {"success": False, "error": "Send button not found"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def forward(self, email: str, to: str) -> Dict[str, Any]:
        """Forward the currently open email."""
        try:
            screenshot = await self._take_screenshot()
            fwd_btn = await self._find_element("Forward button or forward icon", screenshot)
            if not fwd_btn:
                # Try more menu -> Forward
                menu_btn = await self._find_element("More options or three dot menu", screenshot)
                if menu_btn:
                    await self.controller.tap(menu_btn["x"], menu_btn["y"])
                    await asyncio.sleep(0.5)
                    screenshot2 = await self._take_screenshot()
                    fwd_btn = await self._find_element("Forward option", screenshot2)

            if not fwd_btn:
                return {"success": False, "error": "Forward button not found"}

            await self.controller.tap(fwd_btn["x"], fwd_btn["y"])
            await asyncio.sleep(1.5)

            # Type recipient
            await self.controller.type_text(to)
            await self.controller.press_enter()
            await asyncio.sleep(0.5)

            # Send
            screenshot3 = await self._take_screenshot()
            send_btn = await self._find_element("Send button", screenshot3)
            if send_btn:
                await self.controller.tap(send_btn["x"], send_btn["y"])
                await asyncio.sleep(2.0)
                return {"success": True, "action": "forwarded", "to": to}

            return {"success": False, "error": "Send button not found"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Verification ──

    async def wait_for_verification(
        self,
        email: str,
        sender: str = "",
        subject_contains: str = "",
        timeout: float = 120.0,
        poll_interval: float = 10.0,
    ) -> VerificationResult:
        """
        Wait for a verification email and extract the code or link.

        Polls the inbox until a matching email arrives, then extracts
        the verification code (6-digit) or link.

        Args:
            email: Account to check.
            sender: Expected sender (partial match).
            subject_contains: Expected subject text (partial match).
            timeout: Max seconds to wait.
            poll_interval: Seconds between inbox checks.

        Returns:
            VerificationResult with code or link.
        """
        result = VerificationResult(sender=sender)
        start = time.monotonic()

        logger.info("Waiting for verification email to %s from '%s'", email, sender)

        while time.monotonic() - start < timeout:
            messages = await self.read_inbox(email, limit=5)

            for msg in messages:
                # Match by sender
                if sender and sender.lower() not in msg.sender.lower():
                    continue
                # Match by subject
                if subject_contains and subject_contains.lower() not in msg.subject.lower():
                    continue

                # Found a matching email — try to extract code/link
                open_result = await self.open_email(email, msg.sender)
                if not open_result.get("success"):
                    continue

                body = open_result.get("body", "")
                extracted = self._extract_verification(body)
                if extracted.found:
                    extracted.sender = msg.sender
                    extracted.subject = msg.subject
                    self._verification_log.append(extracted)
                    self._save_state()
                    logger.info("Verification found: %s = %s",
                                extracted.verification_type.value,
                                extracted.code or extracted.link[:50])
                    return extracted

                # Go back to inbox
                await self.controller.press_back()
                await asyncio.sleep(1.0)

            logger.debug("No verification email yet, waiting %.0fs...", poll_interval)
            await asyncio.sleep(poll_interval)

        logger.warning("Verification timeout after %.0fs for %s", timeout, email)
        return result

    def _extract_verification(self, text: str) -> VerificationResult:
        """Extract verification code or link from email text."""
        result = VerificationResult()

        if not text:
            return result

        # Look for 6-digit codes
        code_match = re.search(r'\b(\d{6})\b', text)
        if code_match:
            result.found = True
            result.verification_type = VerificationType.CODE
            result.code = code_match.group(1)
            result.raw_text = text[:200]
            return result

        # Look for 4-digit PINs
        pin_match = re.search(r'\b(?:PIN|pin|code)[:\s]*(\d{4})\b', text)
        if pin_match:
            result.found = True
            result.verification_type = VerificationType.PIN
            result.code = pin_match.group(1)
            result.raw_text = text[:200]
            return result

        # Look for verification links
        link_match = re.search(
            r'(https?://\S*(?:verify|confirm|activate|validation|token)\S*)',
            text, re.IGNORECASE
        )
        if link_match:
            result.found = True
            result.verification_type = VerificationType.LINK
            result.link = link_match.group(1)
            result.raw_text = text[:200]
            return result

        return result

    async def complete_verification(self, result: VerificationResult) -> Dict[str, Any]:
        """Complete a verification by entering the code or clicking the link."""
        if not result.found:
            return {"success": False, "error": "No verification to complete"}

        try:
            if result.verification_type in (VerificationType.CODE, VerificationType.PIN):
                # Type the code into the currently focused field
                await self.controller.type_text(result.code)
                await asyncio.sleep(0.5)

                # Try to find and click verify/confirm button
                screenshot = await self._take_screenshot()
                verify_btn = await self._find_element(
                    "Verify button or Confirm button or Submit button", screenshot
                )
                if verify_btn:
                    await self.controller.tap(verify_btn["x"], verify_btn["y"])
                    await asyncio.sleep(2.0)

                return {"success": True, "type": "code", "code": result.code}

            elif result.verification_type == VerificationType.LINK:
                if self.browser:
                    await self.browser.open_url(result.link, wait_for_load=True)
                    await asyncio.sleep(3.0)
                    return {"success": True, "type": "link", "url": result.link}
                else:
                    return {"success": False, "error": "BrowserController not available"}

        except Exception as exc:
            return {"success": False, "error": str(exc)}

        return {"success": False, "error": "Unknown verification type"}

    # ── Account management ──

    def register_account(
        self, email: str, provider: EmailProvider, credential_id: str = "",
        recovery_email: str = "",
    ) -> Dict[str, Any]:
        """Register an existing email account for monitoring."""
        username = email.split("@")[0]
        account = EmailAccount(
            email=email,
            provider=provider,
            username=username,
            credential_id=credential_id,
            recovery_email=recovery_email,
        )
        self._accounts[email] = account
        self._save_state()
        return {"success": True, "account": account.to_dict()}

    def list_accounts(self) -> List[Dict[str, Any]]:
        """List all registered email accounts."""
        return [a.to_dict() for a in self._accounts.values()]

    def remove_account(self, email: str) -> Dict[str, Any]:
        """Remove an account from tracking."""
        if email in self._accounts:
            del self._accounts[email]
            self._messages.pop(email, None)
            self._save_state()
            return {"success": True, "removed": email}
        return {"success": False, "error": f"Account {email} not found"}

    def get_account(self, email: str) -> Optional[Dict[str, Any]]:
        """Get account details."""
        account = self._accounts.get(email)
        return account.to_dict() if account else None

    # ── Multi-account monitoring ──

    async def check_all_accounts(self) -> Dict[str, Any]:
        """Check all accounts for new emails."""
        results = {}
        for email in self._accounts:
            try:
                messages = await self.read_inbox(email, limit=5)
                results[email] = {
                    "messages": len(messages),
                    "unread": sum(1 for m in messages if m.status == EmailStatus.UNREAD),
                }
            except Exception as exc:
                results[email] = {"error": str(exc)}
        return results

    # ── Statistics ──

    def stats(self) -> Dict[str, Any]:
        """Get email agent statistics."""
        return {
            "accounts": len(self._accounts),
            "active": sum(1 for a in self._accounts.values() if a.status == AccountStatus.ACTIVE),
            "total_messages": sum(len(m) for m in self._messages.values()),
            "verifications": len(self._verification_log),
            "providers": {
                p.value: sum(1 for a in self._accounts.values() if a.provider == p)
                for p in EmailProvider if any(a.provider == p for a in self._accounts.values())
            },
        }

    # ── Sync wrappers ──

    def create_gmail_sync(self, username: str, password: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.create_gmail_account(username, password, **kwargs))

    def create_outlook_sync(self, username: str, password: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.create_outlook_account(username, password, **kwargs))

    def read_inbox_sync(self, email: str, **kwargs) -> List[EmailMessage]:
        return _run_sync(self.read_inbox(email, **kwargs))

    def send_sync(self, from_account: str, to: str, subject: str, body: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.compose_and_send(from_account, to, subject, body, **kwargs))

    def verify_sync(self, email: str, **kwargs) -> VerificationResult:
        return _run_sync(self.wait_for_verification(email, **kwargs))

    def login_sync(self, email: str) -> Dict[str, Any]:
        return _run_sync(self.login(email))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[EmailAgent] = None


def get_email_agent(
    controller: Any = None,
    browser: Any = None,
    account_mgr: Any = None,
) -> EmailAgent:
    """Get the singleton EmailAgent instance."""
    global _instance
    if _instance is None:
        _instance = EmailAgent(controller=controller, browser=browser, account_mgr=account_mgr)
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_create(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    provider = args.provider.lower()
    if provider == "gmail":
        result = agent.create_gmail_sync(
            args.username, args.password,
            first_name=args.first_name or "",
            last_name=args.last_name or "",
        )
    elif provider in ("outlook", "hotmail"):
        result = agent.create_outlook_sync(
            args.username, args.password,
            first_name=args.first_name or "",
            last_name=args.last_name or "",
        )
    else:
        result = {"success": False, "error": f"Unsupported provider: {provider}"}
    _print_json(result)


def _cli_inbox(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    messages = agent.read_inbox_sync(args.account, limit=args.limit)
    _print_json([m.to_dict() for m in messages])


def _cli_send(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    result = agent.send_sync(
        args.sender, args.to, args.subject, args.body or ""
    )
    _print_json(result)


def _cli_verify(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    result = agent.verify_sync(
        args.account,
        sender=args.sender or "",
        timeout=args.timeout,
    )
    _print_json(result.to_dict())


def _cli_accounts(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    action = args.action
    if action == "list":
        _print_json(agent.list_accounts())
    elif action == "add":
        provider = EmailProvider(args.provider) if args.provider else EmailProvider.OTHER
        result = agent.register_account(args.email or "", provider)
        _print_json(result)
    elif action == "remove":
        result = agent.remove_account(args.email or "")
        _print_json(result)
    elif action == "check":
        result = _run_sync(agent.check_all_accounts())
        _print_json(result)
    else:
        print(f"Unknown action: {action}")


def _cli_login(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    result = agent.login_sync(args.account)
    _print_json(result)


def _cli_stats(args: argparse.Namespace) -> None:
    agent = get_email_agent()
    _print_json(agent.stats())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="email_agent",
        description="OpenClaw Empire — Email Agent",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # create
    cr = sub.add_parser("create", help="Create an email account")
    cr.add_argument("--provider", required=True, choices=["gmail", "yahoo", "outlook", "hotmail"])
    cr.add_argument("--username", required=True)
    cr.add_argument("--password", required=True)
    cr.add_argument("--first-name", default="")
    cr.add_argument("--last-name", default="")
    cr.set_defaults(func=_cli_create)

    # inbox
    ib = sub.add_parser("inbox", help="Read inbox")
    ib.add_argument("--account", required=True)
    ib.add_argument("--limit", type=int, default=10)
    ib.set_defaults(func=_cli_inbox)

    # send
    sn = sub.add_parser("send", help="Send an email")
    sn.add_argument("--sender", required=True)
    sn.add_argument("--to", required=True)
    sn.add_argument("--subject", required=True)
    sn.add_argument("--body", default="")
    sn.set_defaults(func=_cli_send)

    # verify
    vr = sub.add_parser("verify", help="Wait for verification email")
    vr.add_argument("--account", required=True)
    vr.add_argument("--sender", default="")
    vr.add_argument("--timeout", type=float, default=120.0)
    vr.set_defaults(func=_cli_verify)

    # accounts
    ac = sub.add_parser("accounts", help="Account management")
    ac.add_argument("action", choices=["list", "add", "remove", "check"])
    ac.add_argument("--email", default="")
    ac.add_argument("--provider", default="")
    ac.set_defaults(func=_cli_accounts)

    # login
    lg = sub.add_parser("login", help="Log in to an account")
    lg.add_argument("--account", required=True)
    lg.set_defaults(func=_cli_login)

    # stats
    st = sub.add_parser("stats", help="Email agent statistics")
    st.set_defaults(func=_cli_stats)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
