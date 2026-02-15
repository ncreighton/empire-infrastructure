"""Test email_agent — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.email_agent import (
        EmailAgent,
        EmailAccount,
        EmailMessage,
        VerificationResult,
        ComposeEmail,
        EmailProvider,
        EmailStatus,
        AccountStatus,
        VerificationType,
        PROVIDER_PACKAGES,
        PROVIDER_WEB_URLS,
        SIGNUP_URLS,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="email_agent not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "email"
    d.mkdir()
    return d


@pytest.fixture
def mock_controller():
    ctrl = MagicMock()
    ctrl._adb_shell = AsyncMock(return_value="")
    ctrl.screenshot = AsyncMock(return_value="/tmp/screen.png")
    return ctrl


@pytest.fixture
def mock_browser():
    b = MagicMock()
    b.open_url = AsyncMock(return_value=True)
    b.fill_form = AsyncMock(return_value=True)
    b.extract_page_text = AsyncMock(return_value="Welcome")
    return b


@pytest.fixture
def mock_account_mgr():
    mgr = MagicMock()
    mgr.store_credential = MagicMock(return_value="cred-123")
    return mgr


@pytest.fixture
def agent(data_dir, mock_controller, mock_browser, mock_account_mgr):
    return EmailAgent(
        controller=mock_controller,
        browser=mock_browser,
        account_mgr=mock_account_mgr,
        data_dir=data_dir,
    )


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_email_provider_values(self):
        assert EmailProvider.GMAIL.value == "gmail"
        assert EmailProvider.YAHOO.value == "yahoo"
        assert EmailProvider.OUTLOOK.value == "outlook"
        assert EmailProvider.PROTONMAIL.value == "protonmail"
        assert EmailProvider.OTHER.value == "other"

    def test_email_status_values(self):
        assert EmailStatus.UNREAD.value == "unread"
        assert EmailStatus.READ.value == "read"
        assert EmailStatus.STARRED.value == "starred"
        assert EmailStatus.SPAM.value == "spam"
        assert EmailStatus.DRAFT.value == "draft"

    def test_account_status_values(self):
        assert AccountStatus.ACTIVE.value == "active"
        assert AccountStatus.LOCKED.value == "locked"
        assert AccountStatus.SUSPENDED.value == "suspended"
        assert AccountStatus.NEEDS_VERIFICATION.value == "needs_verification"

    def test_verification_type_values(self):
        assert VerificationType.CODE.value == "code"
        assert VerificationType.LINK.value == "link"
        assert VerificationType.PIN.value == "pin"


# ===================================================================
# Constants Tests
# ===================================================================


class TestConstants:
    def test_provider_packages(self):
        assert EmailProvider.GMAIL in PROVIDER_PACKAGES
        assert "com.google.android.gm" in PROVIDER_PACKAGES[EmailProvider.GMAIL]

    def test_provider_web_urls(self):
        assert EmailProvider.GMAIL in PROVIDER_WEB_URLS
        assert "google" in PROVIDER_WEB_URLS[EmailProvider.GMAIL]

    def test_signup_urls(self):
        assert EmailProvider.GMAIL in SIGNUP_URLS
        assert "accounts.google.com" in SIGNUP_URLS[EmailProvider.GMAIL]


# ===================================================================
# Data Class Tests
# ===================================================================


class TestEmailAccount:
    def test_defaults(self):
        acc = EmailAccount()
        assert acc.email == ""
        assert acc.provider == EmailProvider.OTHER
        assert acc.status == AccountStatus.ACTIVE
        assert acc.unread_count == 0

    def test_to_dict(self):
        acc = EmailAccount(
            email="test@gmail.com",
            provider=EmailProvider.GMAIL,
            status=AccountStatus.ACTIVE,
            unread_count=5,
        )
        d = acc.to_dict()
        assert d["email"] == "test@gmail.com"
        assert d["provider"] == "gmail"
        assert d["status"] == "active"
        assert d["unread_count"] == 5


class TestEmailMessage:
    def test_defaults(self):
        msg = EmailMessage()
        assert msg.sender == ""
        assert msg.status == EmailStatus.UNREAD
        assert msg.has_attachment is False
        assert msg.id != ""

    def test_to_dict(self):
        msg = EmailMessage(
            sender="noreply@instagram.com",
            subject="Verification Code",
            preview="Your code is 123456",
        )
        d = msg.to_dict()
        assert d["sender"] == "noreply@instagram.com"
        assert d["status"] == "unread"

    def test_labels(self):
        msg = EmailMessage(labels=["important", "social"])
        assert "important" in msg.labels
        assert len(msg.labels) == 2


class TestVerificationResult:
    def test_defaults(self):
        vr = VerificationResult()
        assert vr.found is False
        assert vr.verification_type == VerificationType.UNKNOWN
        assert vr.code == ""

    def test_found_code(self):
        vr = VerificationResult(
            found=True,
            verification_type=VerificationType.CODE,
            code="123456",
            sender="noreply@instagram.com",
        )
        assert vr.found is True
        assert vr.code == "123456"

    def test_to_dict(self):
        vr = VerificationResult(
            found=True,
            verification_type=VerificationType.LINK,
            link="https://verify.com/abc",
        )
        d = vr.to_dict()
        assert d["verification_type"] == "link"
        assert d["link"] == "https://verify.com/abc"


class TestComposeEmail:
    def test_defaults(self):
        ce = ComposeEmail()
        assert ce.to == []
        assert ce.subject == ""
        assert ce.attachments == []

    def test_with_data(self):
        ce = ComposeEmail(
            to=["test@example.com"],
            subject="Test",
            body="Hello",
            cc=["cc@example.com"],
        )
        d = ce.to_dict()
        assert "test@example.com" in d["to"]
        assert d["subject"] == "Test"


# ===================================================================
# EmailAgent Tests
# ===================================================================


class TestEmailAgentInit:
    def test_init_creates_data_dir(self, tmp_path, mock_controller, mock_browser, mock_account_mgr):
        d = tmp_path / "new_email"
        agent = EmailAgent(
            controller=mock_controller,
            browser=mock_browser,
            account_mgr=mock_account_mgr,
            data_dir=d,
        )
        assert d.exists()
        assert len(agent._accounts) == 0

    def test_init_empty_state(self, agent):
        assert agent._accounts == {}
        assert agent._messages == {}
        assert agent._verification_log == []


class TestEmailAgentPersistence:
    def test_save_and_load_state(self, data_dir, mock_controller, mock_browser, mock_account_mgr):
        agent = EmailAgent(
            controller=mock_controller,
            browser=mock_browser,
            account_mgr=mock_account_mgr,
            data_dir=data_dir,
        )
        agent._accounts["test@gmail.com"] = EmailAccount(
            email="test@gmail.com",
            provider=EmailProvider.GMAIL,
            status=AccountStatus.ACTIVE,
        )
        agent._messages["test@gmail.com"] = [
            EmailMessage(sender="alice@example.com", subject="Hello"),
        ]
        agent._save_state()

        agent2 = EmailAgent(
            controller=mock_controller,
            browser=mock_browser,
            account_mgr=mock_account_mgr,
            data_dir=data_dir,
        )
        assert "test@gmail.com" in agent2._accounts
        assert agent2._accounts["test@gmail.com"].provider == EmailProvider.GMAIL
        assert len(agent2._messages.get("test@gmail.com", [])) == 1


class TestEmailAgentHelpers:
    @pytest.mark.asyncio
    async def test_adb_shell_delegates(self, agent, mock_controller):
        mock_controller._adb_shell = AsyncMock(return_value="ok")
        result = await agent._adb_shell("pm list packages")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_take_screenshot_delegates(self, agent, mock_controller):
        mock_controller.screenshot = AsyncMock(return_value="/tmp/shot.png")
        result = await agent._take_screenshot()
        assert result == "/tmp/shot.png"


class TestEmailAgentCreateGmail:
    @pytest.mark.asyncio
    async def test_create_gmail_no_browser(self, data_dir, mock_controller, mock_account_mgr):
        agent = EmailAgent(
            controller=mock_controller,
            browser=None,
            account_mgr=mock_account_mgr,
            data_dir=data_dir,
        )
        # Patch the browser property to return None so it doesn't auto-create
        with patch.object(type(agent), "browser", new_callable=lambda: property(lambda self: None)):
            result = await agent.create_gmail_account("john.doe", "Pass123!")
        assert result["success"] is False
        assert "Browser" in result.get("error", "")
