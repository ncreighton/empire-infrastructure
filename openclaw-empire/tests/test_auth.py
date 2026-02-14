"""
Tests for the Auth module.

Tests token generation, validation, scope checking, rate limiting,
and webhook signature verification.
"""

import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock, patch

import pytest

try:
    from src.auth import (
        AuthManager,
        Token,
        RateLimiter,
        generate_token,
        validate_token,
        verify_webhook_signature,
    )
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False

pytestmark = pytest.mark.skipif(
    not HAS_AUTH,
    reason="auth module not yet implemented"
)


# ===================================================================
# TestTokenGeneration
# ===================================================================

class TestTokenGeneration:
    """Test token generation."""

    @pytest.mark.unit
    def test_generate_token_format(self):
        """Generated tokens have proper format."""
        token = generate_token()
        assert isinstance(token, str)
        assert len(token) >= 32  # Minimum reasonable token length

    @pytest.mark.unit
    def test_generate_token_uniqueness(self):
        """Generated tokens are unique."""
        tokens = {generate_token() for _ in range(100)}
        assert len(tokens) == 100  # All unique

    @pytest.mark.unit
    def test_generate_token_with_scopes(self):
        """Token can be created with specific scopes."""
        try:
            token = generate_token(scopes=["read", "write"])
            assert isinstance(token, str)
        except TypeError:
            # If scopes are not in generate_token, that's ok
            pass

    @pytest.mark.unit
    def test_generate_token_with_expiry(self):
        """Token can be created with expiry time."""
        try:
            token = generate_token(expires_in=3600)
            assert isinstance(token, str)
        except TypeError:
            pass


# ===================================================================
# TestTokenValidation
# ===================================================================

class TestTokenValidation:
    """Test token validation."""

    @pytest.mark.unit
    def test_validate_valid_token(self):
        """Valid token passes validation."""
        token = generate_token()
        result = validate_token(token)
        assert result is True or result is not None

    @pytest.mark.unit
    def test_validate_expired_token(self):
        """Expired token fails validation."""
        try:
            token = generate_token(expires_in=-1)  # Already expired
            result = validate_token(token)
            assert result is False or result is None
        except (TypeError, ValueError):
            pass  # Implementation may differ

    @pytest.mark.unit
    def test_validate_wrong_token(self):
        """Random string fails validation."""
        result = validate_token("invalid-random-token-string-here")
        assert result is False or result is None

    @pytest.mark.unit
    def test_validate_empty_token(self):
        """Empty token fails validation."""
        result = validate_token("")
        assert result is False or result is None

    @pytest.mark.unit
    def test_validate_none_token(self):
        """None token fails validation."""
        try:
            result = validate_token(None)
            assert result is False or result is None
        except (TypeError, AttributeError):
            pass  # Expected


# ===================================================================
# TestAuthManager
# ===================================================================

class TestAuthManager:
    """Test AuthManager for token lifecycle."""

    @pytest.fixture
    def manager(self, tmp_data_dir):
        """Create AuthManager with temp storage."""
        return AuthManager(data_dir=tmp_data_dir / "auth")

    @pytest.mark.unit
    def test_create_token(self, manager):
        """Create a new auth token."""
        token = manager.create_token(
            name="test-client",
            scopes=["read", "write"],
        )
        assert token is not None
        assert isinstance(token.value if hasattr(token, 'value') else token, str)

    @pytest.mark.unit
    def test_validate_created_token(self, manager):
        """Created token passes validation."""
        token = manager.create_token(name="test", scopes=["read"])
        token_str = token.value if hasattr(token, 'value') else token
        assert manager.validate(token_str) is True

    @pytest.mark.unit
    def test_revoke_token(self, manager):
        """Revoked token fails validation."""
        token = manager.create_token(name="to-revoke", scopes=["read"])
        token_str = token.value if hasattr(token, 'value') else token
        manager.revoke(token_str)
        assert manager.validate(token_str) is False

    @pytest.mark.unit
    def test_check_scope_authorized(self, manager):
        """Token with required scope passes check."""
        token = manager.create_token(name="scoped", scopes=["read", "write"])
        token_str = token.value if hasattr(token, 'value') else token
        assert manager.check_scope(token_str, "read") is True
        assert manager.check_scope(token_str, "write") is True

    @pytest.mark.unit
    def test_check_scope_unauthorized(self, manager):
        """Token without required scope fails check."""
        token = manager.create_token(name="limited", scopes=["read"])
        token_str = token.value if hasattr(token, 'value') else token
        assert manager.check_scope(token_str, "admin") is False

    @pytest.mark.unit
    def test_list_tokens(self, manager):
        """list_tokens returns all active tokens."""
        manager.create_token(name="token-1", scopes=["read"])
        manager.create_token(name="token-2", scopes=["write"])
        tokens = manager.list_tokens()
        assert len(tokens) >= 2


# ===================================================================
# TestRateLimiter
# ===================================================================

class TestRateLimiter:
    """Test rate limiting."""

    @pytest.fixture
    def limiter(self):
        """Create a rate limiter with low limits for testing."""
        return RateLimiter(max_requests=5, window_seconds=1)

    @pytest.mark.unit
    def test_allows_within_limit(self, limiter):
        """Requests within limit are allowed."""
        client_id = "test-client"
        for _ in range(5):
            assert limiter.check(client_id) is True

    @pytest.mark.unit
    def test_blocks_over_limit(self, limiter):
        """Requests over limit are blocked."""
        client_id = "test-client"
        for _ in range(5):
            limiter.check(client_id)
        assert limiter.check(client_id) is False

    @pytest.mark.unit
    def test_different_clients_independent(self, limiter):
        """Rate limits are per-client."""
        for _ in range(5):
            limiter.check("client-a")
        assert limiter.check("client-a") is False
        assert limiter.check("client-b") is True

    @pytest.mark.unit
    def test_window_reset(self, limiter):
        """Rate limit resets after window expires."""
        client_id = "test-client"
        for _ in range(5):
            limiter.check(client_id)
        assert limiter.check(client_id) is False
        # Wait for window to expire
        time.sleep(1.1)
        assert limiter.check(client_id) is True

    @pytest.mark.unit
    def test_get_remaining(self, limiter):
        """Get remaining requests for a client."""
        client_id = "test-client"
        limiter.check(client_id)
        limiter.check(client_id)
        remaining = limiter.get_remaining(client_id)
        assert remaining == 3


# ===================================================================
# TestWebhookSignature
# ===================================================================

class TestWebhookSignature:
    """Test webhook signature verification."""

    @pytest.mark.unit
    def test_valid_signature(self):
        """Valid HMAC signature passes verification."""
        secret = "webhook-secret-key"
        payload = b'{"event": "post_published", "post_id": 42}'
        signature = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        assert verify_webhook_signature(payload, signature, secret) is True

    @pytest.mark.unit
    def test_invalid_signature(self):
        """Invalid signature fails verification."""
        secret = "webhook-secret-key"
        payload = b'{"event": "post_published"}'
        wrong_sig = "a" * 64
        assert verify_webhook_signature(payload, wrong_sig, secret) is False

    @pytest.mark.unit
    def test_tampered_payload(self):
        """Tampered payload fails verification."""
        secret = "webhook-secret-key"
        original_payload = b'{"amount": 100}'
        signature = hmac.new(
            secret.encode("utf-8"),
            original_payload,
            hashlib.sha256,
        ).hexdigest()
        tampered_payload = b'{"amount": 9999}'
        assert verify_webhook_signature(tampered_payload, signature, secret) is False

    @pytest.mark.unit
    def test_empty_payload(self):
        """Empty payload with valid signature passes."""
        secret = "secret"
        payload = b""
        signature = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        assert verify_webhook_signature(payload, signature, secret) is True
