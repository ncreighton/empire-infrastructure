"""Tests for openclaw/automation/email_verifier.py — IMAP inbox monitoring (no IMAP needed)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from openclaw.automation.email_verifier import (
    EmailVerifier,
    EmailVerifierConfig,
    PLATFORM_EMAIL_SENDERS,
    VERIFICATION_BODY_KEYWORDS,
    VERIFICATION_SUBJECT_PATTERNS,
    VERIFICATION_URL_PATTERNS,
)


@pytest.fixture
def verifier():
    """EmailVerifier with no IMAP configured."""
    config = EmailVerifierConfig(
        imap_host="",
        email_address="",
        email_password="",
    )
    return EmailVerifier(config=config)


@pytest.fixture
def configured_verifier():
    """EmailVerifier with IMAP credentials (but no actual connection)."""
    config = EmailVerifierConfig(
        imap_host="imap.gmail.com",
        imap_port=993,
        email_address="test@example.com",
        email_password="app_password_123",
    )
    return EmailVerifier(config=config)


class TestLoadConfigFromEnv:
    def test_loads_from_env(self):
        with patch.dict(os.environ, {
            "OPENCLAW_IMAP_HOST": "imap.test.com",
            "OPENCLAW_IMAP_PORT": "995",
            "OPENCLAW_EMAIL": "user@test.com",
            "OPENCLAW_EMAIL_PASSWORD": "secret",
        }):
            v = EmailVerifier()
            assert v.config.imap_host == "imap.test.com"
            assert v.config.imap_port == 995
            assert v.config.email_address == "user@test.com"
            assert v.config.email_password == "secret"

    def test_defaults_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            v = EmailVerifier()
            assert v.config.imap_host == "imap.gmail.com"
            assert v.config.imap_port == 993
            assert v.config.email_address == ""
            assert v.config.email_password == ""


class TestIsConfigured:
    def test_not_configured_without_password(self):
        config = EmailVerifierConfig(
            imap_host="imap.gmail.com",
            email_address="test@example.com",
            email_password="",
        )
        v = EmailVerifier(config=config)
        assert v.is_configured is False

    def test_not_configured_without_email(self):
        config = EmailVerifierConfig(
            imap_host="imap.gmail.com",
            email_address="",
            email_password="pass",
        )
        v = EmailVerifier(config=config)
        assert v.is_configured is False

    def test_not_configured_without_host(self):
        config = EmailVerifierConfig(
            imap_host="",
            email_address="test@example.com",
            email_password="pass",
        )
        v = EmailVerifier(config=config)
        assert v.is_configured is False

    def test_is_configured_with_all(self, configured_verifier):
        assert configured_verifier.is_configured is True


class TestDetectPlatform:
    def test_exact_match_gumroad(self, verifier):
        result = verifier._detect_platform("no-reply@gumroad.com")
        assert result == "gumroad"

    def test_exact_match_etsy(self, verifier):
        result = verifier._detect_platform("no-reply@etsy.com")
        assert result == "etsy"

    def test_exact_match_case_insensitive(self, verifier):
        result = verifier._detect_platform("No-Reply@Gumroad.com")
        assert result == "gumroad"

    def test_domain_fallback(self, verifier):
        """Unknown specific address but matching domain should still match."""
        result = verifier._detect_platform("billing@gumroad.com")
        assert result == "gumroad"

    def test_unknown_sender(self, verifier):
        result = verifier._detect_platform("noreply@unknown-platform-xyz.com")
        assert result is None

    def test_all_known_platforms_have_senders(self, verifier):
        """Every platform in PLATFORM_EMAIL_SENDERS should be detectable."""
        for platform_id, senders in PLATFORM_EMAIL_SENDERS.items():
            for sender in senders:
                detected = verifier._detect_platform(sender)
                assert detected == platform_id, f"Failed for {sender} -> expected {platform_id}"


class TestIsVerificationEmail:
    def test_verify_email_subject(self, verifier):
        assert verifier._is_verification_email("Verify your email address", "") is True

    def test_confirm_email_subject(self, verifier):
        assert verifier._is_verification_email("Confirm your email to get started", "") is True

    def test_activate_account_subject(self, verifier):
        assert verifier._is_verification_email("Activate your account", "") is True

    def test_complete_registration_subject(self, verifier):
        assert verifier._is_verification_email("Complete your registration", "") is True

    def test_one_more_step_subject(self, verifier):
        assert verifier._is_verification_email("Just one more step", "") is True

    def test_please_verify_subject(self, verifier):
        assert verifier._is_verification_email("Please verify your email", "") is True

    def test_non_verification_subject(self, verifier):
        assert verifier._is_verification_email("Your weekly newsletter", "") is False

    def test_body_keyword_verify_email(self, verifier):
        body = "Please click the link below to verify your email address."
        assert verifier._is_verification_email("", body) is True

    def test_body_keyword_confirm_email(self, verifier):
        body = "Please confirm your email to continue using our service."
        assert verifier._is_verification_email("", body) is True

    def test_body_keyword_activate_account(self, verifier):
        body = "Click here to activate your account."
        assert verifier._is_verification_email("", body) is True

    def test_unrelated_body(self, verifier):
        body = "Thank you for your purchase. Your order has been shipped."
        assert verifier._is_verification_email("Order Confirmation", body) is False


class TestExtractVerificationUrl:
    def test_extract_from_href(self, verifier):
        body = '<a href="https://example.com/verify?token=abc123">Verify Email</a>'
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert "verify" in url
        assert "token=abc123" in url

    def test_extract_from_confirm_link(self, verifier):
        body = '<a href="https://platform.com/confirm/email/xyz789">Confirm</a>'
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert "confirm" in url

    def test_extract_from_token_param(self, verifier):
        body = '<a href="https://example.com/auth/callback?token=long_token_value_here">Click here</a>'
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert "token=" in url

    def test_skips_unsubscribe_links(self, verifier):
        body = (
            '<a href="https://example.com/unsubscribe?token=abc">Unsubscribe</a>'
            '<a href="https://example.com/verify?token=xyz">Verify</a>'
        )
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert "unsubscribe" not in url
        assert "verify" in url

    def test_skips_image_links(self, verifier):
        body = (
            '<a href="https://example.com/logo.png?token=abc">Logo</a>'
            '<a href="https://example.com/activate?code=xyz">Activate</a>'
        )
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert ".png" not in url

    def test_skips_css_links(self, verifier):
        body = (
            '<a href="https://example.com/styles.css?token=abc">Style</a>'
            '<a href="https://example.com/validate?key=xyz">Validate</a>'
        )
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert ".css" not in url

    def test_plain_text_url(self, verifier):
        body = "Click this link to verify: https://example.com/verify/user/abc123"
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert "verify" in url

    def test_no_verification_url(self, verifier):
        body = "Thank you for your purchase. No links to verify here."
        url = verifier._extract_verification_url(body)
        assert url is None

    def test_html_entity_decoded(self, verifier):
        body = '<a href="https://example.com/verify?token=abc&amp;user=test">Verify</a>'
        url = verifier._extract_verification_url(body)
        assert url is not None
        assert "&amp;" not in url
        assert "&" in url


class TestAddPlatformSender:
    def test_register_new_sender(self, verifier):
        verifier.add_platform_sender("new_platform", "hello@newplatform.com")
        result = verifier._detect_platform("hello@newplatform.com")
        assert result == "new_platform"

    def test_add_to_existing_platform(self, verifier):
        verifier.add_platform_sender("gumroad", "billing@gumroad.com")
        result = verifier._detect_platform("billing@gumroad.com")
        assert result == "gumroad"

    def test_no_duplicate_senders(self, verifier):
        verifier.add_platform_sender("gumroad", "no-reply@gumroad.com")
        # Should not add duplicate
        count = PLATFORM_EMAIL_SENDERS["gumroad"].count("no-reply@gumroad.com")
        assert count == 1


class TestGetStats:
    def test_stats_has_expected_keys(self, verifier):
        stats = verifier.get_stats()
        assert "total_processed" in stats
        assert "verified" in stats
        assert "failed" in stats
        assert "pending_platforms" in stats
        assert "verified_platforms" in stats
        assert "processed_message_ids" in stats
        assert "is_connected" in stats
        assert "is_configured" in stats

    def test_initial_stats_empty(self, verifier):
        stats = verifier.get_stats()
        assert stats["total_processed"] == 0
        assert stats["verified"] == 0
        assert stats["failed"] == 0
        assert stats["pending_platforms"] == []
        assert stats["verified_platforms"] == []

    def test_stats_reflects_not_configured(self, verifier):
        stats = verifier.get_stats()
        assert stats["is_configured"] is False

    def test_stats_reflects_configured(self, configured_verifier):
        stats = configured_verifier.get_stats()
        assert stats["is_configured"] is True

    def test_stats_reflects_not_connected(self, verifier):
        stats = verifier.get_stats()
        assert stats["is_connected"] is False
