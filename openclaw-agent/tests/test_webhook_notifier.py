"""Tests for openclaw/automation/webhook_notifier.py — event notifications."""

import asyncio

import pytest

from openclaw.automation.webhook_notifier import WebhookNotifier, EventType


class TestEventType:
    def test_expected_members_exist(self):
        assert hasattr(EventType, "SIGNUP_STARTED")
        assert hasattr(EventType, "SIGNUP_COMPLETED")
        assert hasattr(EventType, "SIGNUP_FAILED")
        assert hasattr(EventType, "CAPTCHA_NEEDED")
        assert hasattr(EventType, "CAPTCHA_SOLVED")
        assert hasattr(EventType, "EMAIL_VERIFICATION_NEEDED")
        assert hasattr(EventType, "EMAIL_VERIFIED")
        assert hasattr(EventType, "PROFILE_SCORED")
        assert hasattr(EventType, "BATCH_STARTED")
        assert hasattr(EventType, "BATCH_COMPLETED")
        assert hasattr(EventType, "SYNC_COMPLETED")
        assert hasattr(EventType, "ERROR")

    def test_enum_values_are_strings(self):
        for member in EventType:
            assert isinstance(member.value, str)


class TestAddWebhook:
    def test_add_webhook_increases_count(self):
        notifier = WebhookNotifier()
        initial = len(notifier.webhooks)
        notifier.add_webhook("https://hooks.example.com/test", name="test")
        assert len(notifier.webhooks) == initial + 1

    def test_add_multiple_webhooks(self):
        notifier = WebhookNotifier()
        initial = len(notifier.webhooks)
        notifier.add_webhook("https://a.com/hook", name="a")
        notifier.add_webhook("https://b.com/hook", name="b")
        assert len(notifier.webhooks) == initial + 2


class TestEventHistory:
    def test_empty_history_initially(self):
        notifier = WebhookNotifier()
        history = notifier.get_event_history()
        assert isinstance(history, list)
        assert len(history) == 0


class TestGetStats:
    def test_stats_returns_expected_keys(self):
        notifier = WebhookNotifier()
        stats = notifier.get_stats()
        assert isinstance(stats, dict)
        assert "total_events" in stats
        assert "webhooks_configured" in stats
        assert "webhooks_enabled" in stats
        assert "events_by_type" in stats
        assert "recent_errors" in stats

    def test_stats_zero_events_initially(self):
        notifier = WebhookNotifier()
        stats = notifier.get_stats()
        assert stats["total_events"] == 0
        assert stats["recent_errors"] == 0


class TestNotifyMethods:
    """Test that notify methods don't raise even when no webhooks can be reached."""

    def test_notify_signup_started_no_crash(self):
        notifier = WebhookNotifier()
        notifier.add_webhook("http://127.0.0.1:19999/nonexistent", name="test")
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                notifier.notify_signup_started("gumroad", "Gumroad")
            )
            assert isinstance(results, list)
        finally:
            loop.close()

    def test_notify_signup_completed_no_crash(self):
        notifier = WebhookNotifier()
        notifier.add_webhook("http://127.0.0.1:19999/nonexistent", name="test")
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                notifier.notify_signup_completed(
                    "gumroad", "Gumroad",
                    profile_url="https://example.com",
                    score=85.0,
                    duration=10.0,
                )
            )
            assert isinstance(results, list)
        finally:
            loop.close()

    def test_notify_signup_failed_no_crash(self):
        notifier = WebhookNotifier()
        notifier.add_webhook("http://127.0.0.1:19999/nonexistent", name="test")
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                notifier.notify_signup_failed("gumroad", "Gumroad", error="test")
            )
            assert isinstance(results, list)
        finally:
            loop.close()

    def test_event_recorded_in_history(self):
        notifier = WebhookNotifier()
        # No webhooks configured, but notify still records the event
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                notifier.notify(EventType.SIGNUP_STARTED, {"platform_id": "test"})
            )
        finally:
            loop.close()
        history = notifier.get_event_history()
        assert len(history) == 1
        assert history[0]["event_type"] == "signup_started"
