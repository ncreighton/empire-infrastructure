"""Test audit_logger — OpenClaw Empire."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Isolation fixture — redirect data dir before importing
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    """Redirect audit logger data directory to a temp location."""
    monkeypatch.setattr("src.audit_logger.AUDIT_DATA_DIR", tmp_path / "audit")
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    # Reset singleton
    import src.audit_logger as al_mod
    al_mod._audit_logger = None
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.audit_logger import (
    AuditAction,
    AuditCategory,
    AuditEntry,
    AuditLogger,
    AuditSeverity,
    audit,
    get_audit_logger,
    reset_audit_logger,
)


# ===================================================================
# Enum tests
# ===================================================================

class TestAuditEnums:
    """Test AuditAction, AuditCategory, and AuditSeverity enums."""

    def test_audit_action_values(self):
        assert AuditAction.CREATE.value == "create"
        assert AuditAction.READ.value == "read"
        assert AuditAction.UPDATE.value == "update"
        assert AuditAction.DELETE.value == "delete"
        assert AuditAction.EXECUTE.value == "execute"
        assert AuditAction.LOGIN.value == "login"
        assert AuditAction.LOGOUT.value == "logout"
        assert AuditAction.PUBLISH.value == "publish"
        assert AuditAction.GENERATE.value == "generate"
        assert AuditAction.DEPLOY.value == "deploy"
        assert AuditAction.CONFIGURE.value == "configure"
        assert AuditAction.EXPORT.value == "export"
        assert AuditAction.APPROVE.value == "approve"
        assert AuditAction.REJECT.value == "reject"
        assert AuditAction.ESCALATE.value == "escalate"

    def test_audit_category_values(self):
        assert AuditCategory.CONTENT.value == "content"
        assert AuditCategory.WORDPRESS.value == "wordpress"
        assert AuditCategory.PHONE.value == "phone"
        assert AuditCategory.SOCIAL.value == "social"
        assert AuditCategory.REVENUE.value == "revenue"
        assert AuditCategory.AUTH.value == "auth"
        assert AuditCategory.SYSTEM.value == "system"
        assert AuditCategory.AI.value == "ai"
        assert AuditCategory.DEVICE.value == "device"
        assert AuditCategory.PIPELINE.value == "pipeline"
        assert AuditCategory.SCHEDULER.value == "scheduler"
        assert AuditCategory.ACCOUNT.value == "account"

    def test_audit_severity_values(self):
        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"


# ===================================================================
# AuditEntry
# ===================================================================

class TestAuditEntry:
    """Test audit entry creation, serialization, and matching."""

    def test_create_entry_with_all_fields(self):
        entry = AuditEntry(
            id="abc-123",
            timestamp="2026-02-15T10:00:00+00:00",
            action=AuditAction.PUBLISH,
            category=AuditCategory.CONTENT,
            severity=AuditSeverity.INFO,
            actor="content_gen",
            target="witchcraft:post:42",
            operation="Published 'Moon Ritual Guide'",
            module="content_generator",
            success=True,
            duration_ms=3420.5,
            error=None,
            metadata={"word_count": 2800},
            ip_address="10.0.0.1",
            session_id="sess-xyz",
        )
        assert entry.id == "abc-123"
        assert entry.success is True
        assert entry.duration_ms == 3420.5
        assert entry.metadata["word_count"] == 2800

    def test_to_dict_serializes_enums(self):
        entry = AuditEntry(
            id="e1", timestamp="2026-01-01T00:00:00+00:00",
            action=AuditAction.CREATE, category=AuditCategory.AI,
            severity=AuditSeverity.WARNING, actor="a", target="t",
            operation="op", module="mod", success=False,
        )
        d = entry.to_dict()
        assert d["action"] == "create"
        assert d["category"] == "ai"
        assert d["severity"] == "warning"

    def test_from_dict_roundtrip(self):
        entry = AuditEntry(
            id="rt-1", timestamp="2026-02-15T12:00:00+00:00",
            action=AuditAction.DEPLOY, category=AuditCategory.SYSTEM,
            severity=AuditSeverity.CRITICAL, actor="deploy",
            target="vps", operation="deploy v2.1", module="deploy",
            success=True, duration_ms=5000.0,
        )
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.action == AuditAction.DEPLOY
        assert restored.category == AuditCategory.SYSTEM
        assert restored.severity == AuditSeverity.CRITICAL
        assert restored.duration_ms == 5000.0

    def test_from_dict_handles_unknown_enum_values(self):
        d = {"id": "bad", "action": "nonexistent_action", "category": "fake"}
        entry = AuditEntry.from_dict(d)
        # Should fall back to defaults
        assert entry.action == AuditAction.EXECUTE
        assert entry.category == AuditCategory.SYSTEM

    def test_matches_filters_by_action(self):
        entry = AuditEntry(
            id="m1", timestamp="2026-02-15T10:00:00+00:00",
            action=AuditAction.PUBLISH, category=AuditCategory.CONTENT,
            severity=AuditSeverity.INFO, actor="gen", target="site",
            operation="op", module="mod", success=True,
        )
        assert entry.matches(action=AuditAction.PUBLISH) is True
        assert entry.matches(action=AuditAction.DELETE) is False

    def test_matches_filters_by_category(self):
        entry = AuditEntry(
            id="m2", timestamp="2026-02-15T10:00:00+00:00",
            action=AuditAction.CREATE, category=AuditCategory.SOCIAL,
            severity=AuditSeverity.INFO, actor="bot", target="tw",
            operation="tweet", module="social", success=True,
        )
        assert entry.matches(category=AuditCategory.SOCIAL) is True
        assert entry.matches(category=AuditCategory.PHONE) is False

    def test_matches_filters_by_success(self):
        entry = AuditEntry(
            id="m3", timestamp="2026-02-15T10:00:00+00:00",
            action=AuditAction.EXECUTE, category=AuditCategory.SYSTEM,
            severity=AuditSeverity.ERROR, actor="a", target="t",
            operation="op", module="m", success=False,
        )
        assert entry.matches(success=False) is True
        assert entry.matches(success=True) is False


# ===================================================================
# AuditLogger core
# ===================================================================

class TestAuditLogger:
    """Test the AuditLogger class."""

    def test_log_sync_creates_entry(self, tmp_path):
        al = AuditLogger(buffer_size=50, data_dir=tmp_path / "audit")
        entry = al.log_sync(
            action=AuditAction.CREATE,
            category=AuditCategory.CONTENT,
            severity=AuditSeverity.INFO,
            actor="test",
            target="testsite:post:1",
            operation="Created test post",
            module="test",
            success=True,
        )
        assert entry.id is not None
        assert entry.action == AuditAction.CREATE
        assert entry.success is True

    def test_log_sync_with_metadata(self, tmp_path):
        al = AuditLogger(buffer_size=50, data_dir=tmp_path / "audit")
        entry = al.log_sync(
            action=AuditAction.GENERATE,
            category=AuditCategory.AI,
            severity=AuditSeverity.INFO,
            actor="gen",
            target="article",
            operation="Generate content",
            module="ai",
            success=True,
            metadata={"model": "claude-sonnet", "tokens": 1500},
        )
        assert entry.metadata["model"] == "claude-sonnet"
        assert entry.metadata["tokens"] == 1500

    def test_buffer_auto_flushes(self, tmp_path):
        al = AuditLogger(buffer_size=3, data_dir=tmp_path / "audit")
        for i in range(5):
            al.log_sync(
                action=AuditAction.READ,
                category=AuditCategory.SYSTEM,
                severity=AuditSeverity.DEBUG,
                actor="test",
                target=f"target-{i}",
                operation=f"op-{i}",
                module="test",
                success=True,
            )
        # Should have flushed at least once
        assert al._total_flushed > 0

    def test_flush_sync_writes_to_disk(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        al.log_sync(
            action=AuditAction.PUBLISH,
            category=AuditCategory.CONTENT,
            severity=AuditSeverity.INFO,
            actor="pub",
            target="post",
            operation="publish",
            module="content",
            success=True,
        )
        flushed = al.flush_sync()
        assert flushed == 1
        # Check that a JSON file was created
        json_files = list((tmp_path / "audit").glob("*.json"))
        assert len(json_files) > 0

    def test_search_sync_finds_entries(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        al.log_sync(
            action=AuditAction.CREATE,
            category=AuditCategory.CONTENT,
            severity=AuditSeverity.INFO,
            actor="alice",
            target="post:1",
            operation="create post",
            module="content",
            success=True,
        )
        al.log_sync(
            action=AuditAction.DELETE,
            category=AuditCategory.SYSTEM,
            severity=AuditSeverity.WARNING,
            actor="bob",
            target="file:x",
            operation="delete file",
            module="system",
            success=True,
        )
        results = al.search_sync(action=AuditAction.CREATE)
        assert len(results) == 1
        assert results[0].actor == "alice"

    def test_search_by_category(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        al.log_sync(
            action=AuditAction.EXECUTE,
            category=AuditCategory.PHONE,
            severity=AuditSeverity.INFO,
            actor="phone",
            target="device",
            operation="take screenshot",
            module="phone",
            success=True,
        )
        results = al.search_sync(category=AuditCategory.PHONE)
        assert len(results) == 1

    def test_get_daily_summary(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        for _ in range(3):
            al.log_sync(
                action=AuditAction.GENERATE,
                category=AuditCategory.AI,
                severity=AuditSeverity.INFO,
                actor="gen",
                target="article",
                operation="generate",
                module="ai",
                success=True,
                duration_ms=1000.0,
            )
        al.log_sync(
            action=AuditAction.GENERATE,
            category=AuditCategory.AI,
            severity=AuditSeverity.ERROR,
            actor="gen",
            target="article",
            operation="generate",
            module="ai",
            success=False,
            error="API timeout",
        )
        summary = al.get_daily_summary_sync()
        assert summary["total_entries"] == 4
        assert summary["success_count"] == 3
        assert summary["failure_count"] == 1
        assert summary["avg_duration_ms"] is not None

    def test_export_csv(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        al.log_sync(
            action=AuditAction.UPDATE,
            category=AuditCategory.WORDPRESS,
            severity=AuditSeverity.INFO,
            actor="wp",
            target="post:5",
            operation="update post",
            module="wp",
            success=True,
        )
        csv_path = tmp_path / "export.csv"
        count = al.export_csv_sync(csv_path)
        assert count >= 1
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8")
        assert "update" in content

    def test_close_flushes_remaining(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        al.log_sync(
            action=AuditAction.LOGIN,
            category=AuditCategory.AUTH,
            severity=AuditSeverity.INFO,
            actor="user",
            target="system",
            operation="login",
            module="auth",
            success=True,
        )
        assert len(al._entries) == 1
        al.close()
        assert len(al._entries) == 0


# ===================================================================
# @audit decorator
# ===================================================================

class TestAuditDecorator:
    """Test the @audit decorator for automatic function-level auditing."""

    def test_sync_decorator_logs_success(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        with patch("src.audit_logger.get_audit_logger", return_value=al):
            @audit(AuditAction.PUBLISH, AuditCategory.CONTENT, module="publisher")
            def publish_article(site_id: str, target: str = ""):
                return {"published": True}

            result = publish_article("witchcraft", target="post:42")
            assert result["published"] is True

        al.flush_sync()
        results = al.search_sync(action=AuditAction.PUBLISH)
        assert len(results) == 1
        assert results[0].success is True

    def test_sync_decorator_logs_failure(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        with patch("src.audit_logger.get_audit_logger", return_value=al):
            @audit(AuditAction.GENERATE, AuditCategory.AI, module="image_gen")
            def broken_gen(site_id: str):
                raise RuntimeError("generation failed")

            with pytest.raises(RuntimeError, match="generation failed"):
                broken_gen("witchcraft")

        al.flush_sync()
        results = al.search_sync(action=AuditAction.GENERATE)
        assert len(results) == 1
        assert results[0].success is False
        assert "RuntimeError" in results[0].error

    @pytest.mark.asyncio
    async def test_async_decorator_logs_success(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        with patch("src.audit_logger.get_audit_logger", return_value=al):
            @audit(AuditAction.EXECUTE, AuditCategory.DEVICE, module="phone")
            async def take_screenshot(device_id: str):
                return b"image_data"

            result = await take_screenshot("android-1")
            assert result == b"image_data"

        al.flush_sync()
        results = al.search_sync(action=AuditAction.EXECUTE)
        assert len(results) == 1
        assert results[0].success is True


# ===================================================================
# Context manager
# ===================================================================

class TestAuditOperation:
    """Test the audit_operation context manager."""

    def test_success_logs_entry(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        with al.audit_operation(
            action=AuditAction.DEPLOY,
            category=AuditCategory.SYSTEM,
            actor="deploy_script",
            target="contabo-vps",
            module="deploy",
        ) as ctx:
            ctx.metadata["version"] = "2.1.0"
            ctx.operation = "Deploy v2.1.0"

        al.flush_sync()
        results = al.search_sync(action=AuditAction.DEPLOY)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].duration_ms is not None
        assert results[0].duration_ms >= 0

    def test_failure_logs_error(self, tmp_path):
        al = AuditLogger(buffer_size=100, data_dir=tmp_path / "audit")
        with pytest.raises(ValueError, match="deploy exploded"):
            with al.audit_operation(
                action=AuditAction.DEPLOY,
                category=AuditCategory.SYSTEM,
                actor="deploy",
                target="vps",
                module="deploy",
            ) as ctx:
                raise ValueError("deploy exploded")

        al.flush_sync()
        results = al.search_sync(action=AuditAction.DEPLOY)
        assert len(results) == 1
        assert results[0].success is False
        assert "ValueError" in results[0].error


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_audit_logger_returns_same_instance(self):
        a1 = get_audit_logger()
        a2 = get_audit_logger()
        assert a1 is a2

    def test_reset_audit_logger_clears_singleton(self):
        a1 = get_audit_logger()
        reset_audit_logger()
        a2 = get_audit_logger()
        assert a1 is not a2
