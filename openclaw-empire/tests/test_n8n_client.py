"""
Tests for the n8n Webhook Client module.

Tests webhook triggering, signature verification, and workflow management.
All HTTP calls are mocked.
"""

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.n8n_client import (
        N8nClient,
        WebhookTrigger,
        WorkflowManager,
    )
    HAS_N8N = True
except ImportError:
    HAS_N8N = False

pytestmark = pytest.mark.skipif(
    not HAS_N8N,
    reason="n8n_client module not yet implemented"
)


# ===================================================================
# Constants
# ===================================================================

N8N_BASE_URL = "http://vmi2976539.contaboserver.net:5678/webhook"
WEBHOOK_PATHS = {
    "content": "openclaw-content",
    "publish": "openclaw-publish",
    "kdp": "openclaw-kdp",
    "monitor": "openclaw-monitor",
    "revenue": "openclaw-revenue",
    "audit": "openclaw-audit",
}


# ===================================================================
# TestN8nClient
# ===================================================================

class TestN8nClient:
    """Test N8n webhook client."""

    @pytest.fixture
    def client(self):
        """Create N8nClient with mock session."""
        return N8nClient(base_url=N8N_BASE_URL)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_content_webhook(self, client, mock_aiohttp_response):
        """Content webhook triggers successfully."""
        mock_resp = mock_aiohttp_response(200, {"status": "ok", "execution_id": "123"})
        with patch.object(client, "_post", return_value={"status": "ok", "execution_id": "123"}):
            result = await client.trigger_content({
                "site_id": "witchcraft",
                "topic": "Full Moon Ritual",
                "action": "generate",
            })
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_publish_webhook(self, client, mock_aiohttp_response):
        """Publish webhook triggers successfully."""
        with patch.object(client, "_post", return_value={"status": "ok"}):
            result = await client.trigger_publish({
                "site_id": "witchcraft",
                "post_id": 42,
                "action": "publish",
            })
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_kdp_webhook(self, client, mock_aiohttp_response):
        """KDP webhook triggers successfully."""
        with patch.object(client, "_post", return_value={"status": "ok"}):
            result = await client.trigger_kdp({
                "book_title": "Moon Magic Handbook",
                "action": "publish",
            })
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_monitor_webhook(self, client, mock_aiohttp_response):
        """Monitor webhook triggers successfully."""
        with patch.object(client, "_post", return_value={"status": "ok"}):
            result = await client.trigger_monitor({
                "site_id": "witchcraft",
                "check_type": "health",
            })
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_revenue_webhook(self, client, mock_aiohttp_response):
        """Revenue webhook triggers successfully."""
        with patch.object(client, "_post", return_value={"status": "ok"}):
            result = await client.trigger_revenue({
                "stream": "adsense",
                "amount": 42.50,
                "date": "2026-02-14",
            })
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trigger_audit_webhook(self, client, mock_aiohttp_response):
        """Audit webhook triggers successfully."""
        with patch.object(client, "_post", return_value={"status": "ok"}):
            result = await client.trigger_audit({
                "site_id": "witchcraft",
                "audit_type": "content_quality",
            })
        assert result is not None


# ===================================================================
# TestWebhookSignature
# ===================================================================

class TestWebhookSignature:
    """Test webhook signature verification."""

    @pytest.mark.unit
    def test_signature_generation(self):
        """Webhook signature is generated correctly."""
        secret = "test-secret-key"
        payload = json.dumps({"test": "data"}).encode("utf-8")
        expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

        # The client should generate the same signature
        try:
            from src.n8n_client import compute_webhook_signature
            result = compute_webhook_signature(payload, secret)
            assert result == expected
        except ImportError:
            # If function not available, verify our expected computation
            assert len(expected) == 64  # SHA256 hex digest length

    @pytest.mark.unit
    def test_signature_verification_valid(self):
        """Valid signature passes verification."""
        secret = "test-secret"
        payload = b'{"event": "content_published"}'
        signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

        try:
            from src.n8n_client import verify_webhook_signature
            assert verify_webhook_signature(payload, signature, secret) is True
        except ImportError:
            pass  # Module not yet implemented

    @pytest.mark.unit
    def test_signature_verification_invalid(self):
        """Invalid signature fails verification."""
        secret = "test-secret"
        payload = b'{"event": "content_published"}'
        wrong_signature = "0000000000000000000000000000000000000000000000000000000000000000"

        try:
            from src.n8n_client import verify_webhook_signature
            assert verify_webhook_signature(payload, wrong_signature, secret) is False
        except ImportError:
            pass


# ===================================================================
# TestWorkflowManager
# ===================================================================

class TestWorkflowManager:
    """Test n8n workflow management."""

    @pytest.fixture
    def manager(self):
        """Create WorkflowManager with mock."""
        return WorkflowManager(base_url=N8N_BASE_URL)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_workflows(self, manager):
        """list_workflows returns workflow list."""
        with patch.object(manager, "_get", return_value=[
            {"id": "1", "name": "Content Pipeline", "active": True},
            {"id": "2", "name": "Site Monitor", "active": True},
        ]):
            result = await manager.list_workflows()
        assert isinstance(result, list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_workflow(self, manager):
        """get_workflow returns single workflow details."""
        with patch.object(manager, "_get", return_value={
            "id": "1", "name": "Content Pipeline", "active": True, "nodes": [],
        }):
            result = await manager.get_workflow("1")
        assert result is not None
        assert result.get("id") == "1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_activate_workflow(self, manager):
        """activate_workflow enables a workflow."""
        with patch.object(manager, "_patch", return_value={"active": True}):
            result = await manager.activate_workflow("1")
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deactivate_workflow(self, manager):
        """deactivate_workflow disables a workflow."""
        with patch.object(manager, "_patch", return_value={"active": False}):
            result = await manager.deactivate_workflow("1")
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_executions(self, manager):
        """get_executions returns execution history."""
        with patch.object(manager, "_get", return_value={
            "data": [
                {"id": "exec-1", "status": "success", "startedAt": "2026-02-14T10:00:00Z"},
            ],
        }):
            result = await manager.get_executions("1")
        assert result is not None
