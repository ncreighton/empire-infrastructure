"""Test encryption_layer — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    """Redirect encryption data to temp dir and set master key."""
    enc_dir = tmp_path / "encryption"
    enc_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.encryption_layer.ENCRYPTION_DATA_DIR", enc_dir)
    monkeypatch.setattr("src.encryption_layer.KEYS_FILE", enc_dir / "keys.json.enc")
    monkeypatch.setattr("src.encryption_layer.SECURE_CONFIG_FILE", enc_dir / "secure_config.enc")
    monkeypatch.setattr("src.encryption_layer.MASTER_KEY_FILE", enc_dir / ".master_key")
    monkeypatch.setattr("src.encryption_layer.ROTATION_LOG_FILE", enc_dir / "rotation_log.json")
    # Set a deterministic master key for tests
    monkeypatch.setenv("OPENCLAW_MASTER_KEY", "test-master-key-for-unit-tests-2026")
    # Reset singletons
    import src.encryption_layer as enc_mod
    enc_mod._encryption_layer = None
    enc_mod._secure_config = None
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.encryption_layer import (
    EncryptionAlgorithm,
    EncryptionLayer,
    KeyInfo,
    KeyManager,
    KeyStatus,
    SecureConfig,
    get_encryption_layer,
    get_secure_config,
    encrypt,
    decrypt,
)


# ===================================================================
# Enum tests
# ===================================================================

class TestEncryptionEnums:
    def test_algorithm_values(self):
        assert EncryptionAlgorithm.FERNET.value == "fernet"
        assert EncryptionAlgorithm.AES_GCM.value == "aes_gcm"

    def test_key_status_values(self):
        assert KeyStatus.ACTIVE.value == "active"
        assert KeyStatus.ROTATED.value == "rotated"
        assert KeyStatus.REVOKED.value == "revoked"
        assert KeyStatus.EXPIRED.value == "expired"


# ===================================================================
# KeyInfo
# ===================================================================

class TestKeyInfo:
    def test_to_dict_and_from_dict_roundtrip(self):
        info = KeyInfo(
            key_id="abc-123",
            algorithm=EncryptionAlgorithm.FERNET,
            created_at="2026-01-01T00:00:00+00:00",
            rotated_at=None,
            expires_at="2027-01-01T00:00:00+00:00",
            status=KeyStatus.ACTIVE,
            version=1,
            description="Test key",
        )
        d = info.to_dict()
        assert d["algorithm"] == "fernet"
        assert d["status"] == "active"

        restored = KeyInfo.from_dict(d)
        assert restored.key_id == "abc-123"
        assert restored.algorithm == EncryptionAlgorithm.FERNET
        assert restored.status == KeyStatus.ACTIVE

    def test_is_usable_for_active_key(self):
        info = KeyInfo(
            key_id="a", algorithm=EncryptionAlgorithm.FERNET,
            created_at="", rotated_at=None, expires_at=None,
            status=KeyStatus.ACTIVE, version=1, description="",
        )
        assert info.is_usable is True

    def test_is_usable_for_rotated_key(self):
        info = KeyInfo(
            key_id="r", algorithm=EncryptionAlgorithm.FERNET,
            created_at="", rotated_at="2026-01-01", expires_at=None,
            status=KeyStatus.ROTATED, version=1, description="",
        )
        assert info.is_usable is True

    def test_is_not_usable_for_revoked_key(self):
        info = KeyInfo(
            key_id="rev", algorithm=EncryptionAlgorithm.FERNET,
            created_at="", rotated_at=None, expires_at=None,
            status=KeyStatus.REVOKED, version=1, description="",
        )
        assert info.is_usable is False


# ===================================================================
# KeyManager
# ===================================================================

class TestKeyManager:
    def test_generate_key_creates_active_key(self):
        km = KeyManager()
        info = km.generate_key(description="unit test key")
        assert info.status == KeyStatus.ACTIVE
        assert info.version == 1
        assert info.algorithm == EncryptionAlgorithm.FERNET

    def test_get_active_key_auto_generates(self):
        km = KeyManager()
        key_id, key_bytes = km.get_active_key()
        assert len(key_id) > 0
        assert len(key_bytes) > 0

    def test_generate_key_rotates_previous(self):
        km = KeyManager()
        first = km.generate_key(description="first")
        second = km.generate_key(description="second")
        assert second.version == 2
        keys = km.list_keys()
        statuses = {k.key_id: k.status for k in keys}
        assert statuses[first.key_id] == KeyStatus.ROTATED
        assert statuses[second.key_id] == KeyStatus.ACTIVE

    def test_rotate_key(self):
        km = KeyManager()
        km.generate_key(description="initial")
        new_info = km.rotate_key(description="rotated")
        assert new_info.status == KeyStatus.ACTIVE
        assert new_info.version == 2

    def test_revoke_key_non_active(self):
        km = KeyManager()
        first = km.generate_key(description="first")
        km.generate_key(description="second")
        result = km.revoke_key(first.key_id)
        assert result is True
        keys = km.list_keys()
        revoked = [k for k in keys if k.key_id == first.key_id]
        assert revoked[0].status == KeyStatus.REVOKED

    def test_revoke_active_key_fails(self):
        km = KeyManager()
        info = km.generate_key(description="active")
        result = km.revoke_key(info.key_id)
        assert result is False

    def test_has_keys(self):
        km = KeyManager()
        assert km.has_keys() is False
        km.generate_key()
        assert km.has_keys() is True

    def test_list_keys_sorted_by_version_desc(self):
        km = KeyManager()
        km.generate_key(description="v1")
        km.generate_key(description="v2")
        km.generate_key(description="v3")
        keys = km.list_keys()
        versions = [k.version for k in keys]
        assert versions == sorted(versions, reverse=True)


# ===================================================================
# EncryptionLayer — data encryption
# ===================================================================

class TestEncryptionLayerData:
    """Test in-memory encrypt/decrypt round-trips."""

    def test_encrypt_decrypt_string_roundtrip(self):
        enc = EncryptionLayer()
        plaintext = "Hello secret world!"
        ciphertext = enc.encrypt_data_sync(plaintext)
        assert "OPENCLAW_ENC_V1" in ciphertext
        result = enc.decrypt_data_sync(ciphertext)
        assert result == plaintext

    def test_encrypt_decrypt_dict_roundtrip(self):
        enc = EncryptionLayer()
        data = {"api_key": "sk-secret-123", "count": 42}
        ciphertext = enc.encrypt_data_sync(data)
        result = enc.decrypt_data_sync(ciphertext)
        assert isinstance(result, dict)
        assert result["api_key"] == "sk-secret-123"
        assert result["count"] == 42

    def test_encrypt_decrypt_bytes_roundtrip(self):
        enc = EncryptionLayer()
        plaintext = b"raw binary data \x00\x01\x02"
        ciphertext = enc.encrypt_data_sync(plaintext)
        result = enc.decrypt_data_sync(ciphertext)
        # Bytes are decoded to string since they are valid UTF-8...
        # but raw bytes with nulls won't be valid JSON, so returned as str
        assert isinstance(result, str)

    def test_decrypt_invalid_format_raises(self):
        enc = EncryptionLayer()
        with pytest.raises(ValueError, match="Invalid encrypted data"):
            enc.decrypt_data_sync("not-encrypted-data-at-all")

    def test_decrypt_wrong_header_raises(self):
        enc = EncryptionLayer()
        with pytest.raises(ValueError, match="Invalid encryption header"):
            enc.decrypt_data_sync("WRONG_HEADER|key|algo\ndata")


# ===================================================================
# EncryptionLayer — file encryption
# ===================================================================

class TestEncryptionLayerFile:
    """Test file-level encryption and decryption."""

    def test_encrypt_and_decrypt_file(self, tmp_path):
        enc = EncryptionLayer()
        # Create a plaintext file
        plain_file = tmp_path / "secret.json"
        data = {"password": "hunter2", "db": "prod"}
        plain_file.write_text(json.dumps(data), encoding="utf-8")

        # Encrypt
        enc_path = enc.encrypt_file_sync(plain_file)
        assert enc_path.suffix == ".enc"
        assert enc_path.exists()
        assert not plain_file.exists()  # original is securely deleted

        # Decrypt
        dec_path = enc.decrypt_file_sync(enc_path)
        assert dec_path.exists()
        assert not enc_path.exists()  # encrypted is securely deleted
        restored = json.loads(dec_path.read_text(encoding="utf-8"))
        assert restored["password"] == "hunter2"

    def test_encrypt_already_enc_skips(self, tmp_path):
        enc = EncryptionLayer()
        enc_file = tmp_path / "already.json.enc"
        enc_file.write_text("dummy", encoding="utf-8")
        result = enc.encrypt_file_sync(enc_file)
        assert result == enc_file


# ===================================================================
# EncryptionLayer — JSON encryption
# ===================================================================

class TestEncryptionLayerJson:
    """Test encrypted JSON file operations."""

    def test_encrypt_json_and_decrypt_json(self, tmp_path):
        enc = EncryptionLayer()
        data = {"credentials": {"user": "admin", "pass": "s3cret"}}
        out_path = tmp_path / "creds.enc"
        enc.encrypt_json_sync(data, out_path)
        assert out_path.exists()

        restored = enc.decrypt_json_sync(out_path)
        assert restored["credentials"]["user"] == "admin"
        assert restored["credentials"]["pass"] == "s3cret"

    def test_decrypt_json_missing_file_raises(self, tmp_path):
        enc = EncryptionLayer()
        with pytest.raises(FileNotFoundError):
            enc.decrypt_json_sync(tmp_path / "nonexistent.enc")


# ===================================================================
# Key rotation
# ===================================================================

class TestKeyRotation:
    def test_key_rotation_preserves_data(self):
        enc = EncryptionLayer()
        # Encrypt with key v1
        original = {"secret": "v1-data"}
        ciphertext = enc.encrypt_data_sync(original)

        # Rotate key
        enc.key_manager.rotate_key(description="test rotation")

        # Old ciphertext should still be decryptable
        result = enc.decrypt_data_sync(ciphertext)
        assert result["secret"] == "v1-data"

        # New data should use new key
        new_cipher = enc.encrypt_data_sync({"secret": "v2-data"})
        new_result = enc.decrypt_data_sync(new_cipher)
        assert new_result["secret"] == "v2-data"


# ===================================================================
# SecureConfig
# ===================================================================

class TestSecureConfig:
    """Test the encrypted config key-value store."""

    def test_set_and_get(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("api_key", "sk-test-123")
        assert cfg.get("api_key") == "sk-test-123"

    def test_get_default_for_missing_key(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        assert cfg.get("nonexistent", "default_val") == "default_val"

    def test_delete_key(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("temp_key", "temp_value")
        assert cfg.delete("temp_key") is True
        assert cfg.get("temp_key") is None
        assert cfg.delete("temp_key") is False

    def test_list_keys(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("alpha", "a")
        cfg.set("beta", "b")
        keys = cfg.list_keys()
        assert "alpha" in keys
        assert "beta" in keys

    def test_has(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("exists", "yes")
        assert cfg.has("exists") is True
        assert cfg.has("nope") is False

    def test_count(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("k1", "v1")
        cfg.set("k2", "v2")
        assert cfg.count() == 2

    def test_clear(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("k1", "v1")
        cfg.clear()
        assert cfg.count() == 0

    def test_underscore_key_rejected(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        with pytest.raises(ValueError, match="cannot start with underscore"):
            cfg.set("_internal", "bad")

    def test_export_redacted(self):
        enc = EncryptionLayer()
        cfg = SecureConfig(encryption=enc)
        cfg.set("long_secret", "abcdefghijklmnop")
        redacted = cfg.export_redacted()
        assert "***" in redacted["long_secret"]
        # First 3 and last 3 chars visible
        assert redacted["long_secret"].startswith("abc")
        assert redacted["long_secret"].endswith("nop")


# ===================================================================
# Singleton
# ===================================================================

class TestSingleton:
    def test_get_encryption_layer_returns_same_instance(self):
        e1 = get_encryption_layer()
        e2 = get_encryption_layer()
        assert e1 is e2

    def test_get_secure_config_returns_same_instance(self):
        c1 = get_secure_config()
        c2 = get_secure_config()
        assert c1 is c2


# ===================================================================
# Convenience functions
# ===================================================================

class TestConvenienceFunctions:
    def test_encrypt_and_decrypt_convenience(self):
        ciphertext = encrypt("hello from convenience")
        result = decrypt(ciphertext)
        assert result == "hello from convenience"

    def test_encrypt_dict_convenience(self):
        data = {"key": "value"}
        ciphertext = encrypt(data)
        result = decrypt(ciphertext)
        assert isinstance(result, dict)
        assert result["key"] == "value"
