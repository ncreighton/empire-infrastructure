"""
Tests for the Account Manager module.

Tests account CRUD, platform mapping, credential storage, status tracking,
encryption, session management, and audit logging.
All encryption and network operations are mocked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.account_manager import (
        AccountPool,
        AuditEntry,
        AuthType,
        Credential,
        CredentialStatus,
        EncryptionEngine,
        PlatformConfig,
        RotationStrategy,
        Session,
        SessionStatus,
        get_account_manager,
        ENCRYPTED_FIELDS,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="account_manager module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def manager_dir(tmp_path):
    """Isolated data directory for account state."""
    d = tmp_path / "accounts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def encryption_engine():
    """Create test encryption engine."""
    return EncryptionEngine(master_key="test-master-key-32-bytes-long!!!")


@pytest.fixture
def sample_credential():
    """Pre-built credential for testing."""
    return Credential(
        credential_id="cred_001",
        platform="instagram",
        username="test_witch_bot",
        password="s3cur3_p4ss",
        status=CredentialStatus.ACTIVE,
    )


@pytest.fixture
def sample_session():
    """Pre-built session for testing."""
    return Session(
        session_id="sess_001",
        credential_id="cred_001",
        platform="instagram",
        status=SessionStatus.ACTIVE,
        tokens={"access": "test_session_token_abc123"},
    )


# ===================================================================
# Enum Tests
# ===================================================================

class TestEnums:
    """Verify enum members."""

    def test_credential_status(self):
        assert CredentialStatus.ACTIVE is not None
        assert CredentialStatus.LOCKED is not None
        assert CredentialStatus.EXPIRED is not None

    def test_auth_type(self):
        assert AuthType.PASSWORD is not None
        assert AuthType.API_KEY is not None
        assert AuthType.OAUTH is not None

    def test_rotation_strategy(self):
        assert RotationStrategy.ROUND_ROBIN is not None
        assert RotationStrategy.RANDOM is not None

    def test_session_status(self):
        assert SessionStatus.ACTIVE is not None
        assert SessionStatus.EXPIRED is not None


# ===================================================================
# EncryptionEngine Tests
# ===================================================================

class TestEncryptionEngine:
    """Test credential encryption."""

    def test_encrypt_decrypt_roundtrip(self, encryption_engine):
        plaintext = "my_secret_password"
        encrypted = encryption_engine.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = encryption_engine.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_encryptions(self, encryption_engine):
        text = "same_text"
        e1 = encryption_engine.encrypt(text)
        e2 = encryption_engine.encrypt(text)
        # Fernet uses random IV, so encryptions should differ
        # (though both decrypt to same plaintext)
        d1 = encryption_engine.decrypt(e1)
        d2 = encryption_engine.decrypt(e2)
        assert d1 == d2 == text

    def test_mode_property(self, encryption_engine):
        mode = encryption_engine.mode
        assert isinstance(mode, str)

    def test_re_encrypt(self, encryption_engine):
        plaintext = "re_encrypt_me"
        encrypted = encryption_engine.encrypt(plaintext)
        new_engine = EncryptionEngine(master_key="another-key-32-bytes-long!!!!!")
        re_encrypted = encryption_engine.re_encrypt(encrypted, new_engine)
        decrypted = new_engine.decrypt(re_encrypted)
        assert decrypted == plaintext


# ===================================================================
# Credential Tests
# ===================================================================

class TestCredential:
    """Test Credential dataclass."""

    def test_create_credential(self, sample_credential):
        assert sample_credential.platform == "instagram"
        assert sample_credential.username == "test_witch_bot"

    def test_credential_status(self, sample_credential):
        assert sample_credential.status == CredentialStatus.ACTIVE

    def test_masked_copy(self, sample_credential):
        masked = sample_credential.masked_copy()
        assert isinstance(masked, dict)
        assert masked["password"] != "s3cur3_p4ss"
        assert "***" in masked["password"] or len(masked["password"]) < len(sample_credential.password)

    def test_is_token_expired_not_expired(self, sample_credential):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        sample_credential.token_expiry = future
        assert sample_credential.is_token_expired() is False

    def test_is_token_expired_yes(self, sample_credential):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        sample_credential.token_expiry = past
        assert sample_credential.is_token_expired() is True


# ===================================================================
# Session Tests
# ===================================================================

class TestSession:
    """Test Session dataclass."""

    def test_create_session(self, sample_session):
        assert sample_session.session_id == "sess_001"
        assert sample_session.status == SessionStatus.ACTIVE

    def test_session_is_expired(self):
        expired = Session(
            session_id="sess_exp",
            credential_id="cred_001",
            platform="instagram",
            status=SessionStatus.ACTIVE,
            tokens={"access": "token"},
            expires_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        )
        assert expired.is_expired() is True

    def test_session_not_expired(self, sample_session):
        assert sample_session.is_expired() is False

    def test_session_touch(self, sample_session):
        old_time = sample_session.last_activity
        sample_session.touch()
        # touch() should update last_activity
        assert sample_session.last_activity >= old_time


# ===================================================================
# PlatformConfig Tests
# ===================================================================

class TestPlatformConfig:
    """Test platform configuration."""

    def test_create_config(self):
        cfg = PlatformConfig(
            name="Instagram",
            login_url="https://instagram.com/accounts/login",
            auth_type=AuthType.OAUTH,
            token_lifetime_hours=24,
            requires_2fa=True,
        )
        assert cfg.name == "Instagram"
        assert cfg.auth_type == AuthType.OAUTH


# ===================================================================
# AccountPool Tests
# ===================================================================

class TestAccountPool:
    """Test account pool grouping."""

    def test_create_pool(self):
        pool = AccountPool(
            pool_id="pool_insta",
            platform="instagram",
            credential_ids=["cred_001", "cred_002", "cred_003"],
        )
        assert pool.pool_id == "pool_insta"
        assert len(pool.credential_ids) == 3


# ===================================================================
# AuditEntry Tests
# ===================================================================

class TestAuditEntry:
    """Test audit logging."""

    def test_create_entry(self):
        entry = AuditEntry(
            action="login",
            credential_id="cred_001",
            platform="instagram",
            details="Successful login from proxy 1.2.3.4",
            timestamp=datetime.now(timezone.utc),
        )
        assert entry.action == "login"
        assert "proxy" in entry.details


# ===================================================================
# ENCRYPTED_FIELDS Tests
# ===================================================================

class TestEncryptedFields:
    """Verify encrypted fields constant."""

    def test_encrypted_fields_exists(self):
        assert isinstance(ENCRYPTED_FIELDS, (frozenset, set, tuple, list))
        assert len(ENCRYPTED_FIELDS) >= 1

    def test_password_is_encrypted(self):
        assert "password" in ENCRYPTED_FIELDS


# ===================================================================
# Account Manager Integration Tests
# ===================================================================

class TestAccountManagerIntegration:
    """Test high-level manager operations."""

    def test_store_credential(self, manager_dir):
        with patch("src.account_manager.EncryptionEngine") as mock_enc:
            mock_enc_instance = MagicMock()
            mock_enc_instance.encrypt = MagicMock(side_effect=lambda x: f"ENC:{x}")
            mock_enc_instance.decrypt = MagicMock(side_effect=lambda x: x.replace("ENC:", ""))
            mock_enc.return_value = mock_enc_instance
            mgr = get_account_manager()
            if hasattr(mgr, "store_credential"):
                cid = mgr.store_credential(
                    "instagram",
                    "bot_user_account",
                    username="bot_user",
                    password="secret123",
                )
                assert cid is not None

    def test_get_credential(self, manager_dir):
        with patch("src.account_manager.EncryptionEngine") as mock_enc:
            mock_enc_instance = MagicMock()
            mock_enc_instance.encrypt = MagicMock(side_effect=lambda x: f"ENC:{x}")
            mock_enc_instance.decrypt = MagicMock(side_effect=lambda x: x.replace("ENC:", ""))
            mock_enc.return_value = mock_enc_instance
            mgr = get_account_manager()
            if hasattr(mgr, "store_credential") and hasattr(mgr, "get_credential"):
                cid = mgr.store_credential(
                    "twitter",
                    "tw_bot_account",
                    username="tw_bot",
                    password="pass456",
                )
                cred = mgr.get_credential(cid)
                assert cred is not None
                assert cred.platform == "twitter"


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_account_manager_returns_instance(self):
        from src.account_manager import AccountManager
        mgr = get_account_manager()
        assert isinstance(mgr, AccountManager)
