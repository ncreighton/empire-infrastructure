"""Tests for openclaw/security/credential_store.py — Fernet encryption."""

import pytest

from openclaw.security.credential_store import CredentialStore


@pytest.fixture
def store(tmp_path):
    """Create a CredentialStore with a temporary key file."""
    key_path = str(tmp_path / "test_encryption.key")
    return CredentialStore(key_path=key_path)


class TestCredentialStore:
    def test_encrypt_decrypt_roundtrip(self, store):
        plaintext = "my-secret-password-123!"
        encrypted = store.encrypt(plaintext)
        decrypted = store.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_produces_different_ciphertext(self, store):
        """Fernet produces different ciphertext each time (includes timestamp/IV)."""
        plaintext = "same-input"
        c1 = store.encrypt(plaintext)
        c2 = store.encrypt(plaintext)
        assert c1 != c2  # Fernet tokens differ due to timestamp

    def test_ciphertext_is_not_plaintext(self, store):
        plaintext = "visible-secret"
        encrypted = store.encrypt(plaintext)
        assert plaintext not in encrypted

    def test_encrypt_dict_decrypt_dict_roundtrip(self, store):
        data = {
            "email": "user@example.com",
            "password": "s3cr3t!",
            "token": "abc-def-ghi",
            "nested": {"key": "value"},
        }
        encrypted = store.encrypt_dict(data)
        decrypted = store.decrypt_dict(encrypted)
        assert decrypted == data

    def test_encrypt_dict_empty(self, store):
        encrypted = store.encrypt_dict({})
        decrypted = store.decrypt_dict(encrypted)
        assert decrypted == {}

    def test_generate_key_format(self):
        key = CredentialStore.generate_key()
        assert isinstance(key, str)
        # Fernet keys are 44 bytes base64-encoded
        assert len(key) == 44
        assert key.endswith("=")

    def test_generate_key_unique(self):
        k1 = CredentialStore.generate_key()
        k2 = CredentialStore.generate_key()
        assert k1 != k2

    def test_key_persistence(self, tmp_path):
        """Key should be created on first use and reused on subsequent instances."""
        key_path = str(tmp_path / "persist_key.key")
        store1 = CredentialStore(key_path=key_path)
        encrypted = store1.encrypt("hello")

        # Create a new instance pointing to the same key file
        store2 = CredentialStore(key_path=key_path)
        decrypted = store2.decrypt(encrypted)
        assert decrypted == "hello"

    def test_unicode_roundtrip(self, store):
        plaintext = "password-with-unicode-\u00e9\u00e8\u00ea-\u4e16\u754c"
        encrypted = store.encrypt(plaintext)
        decrypted = store.decrypt(encrypted)
        assert decrypted == plaintext

    def test_long_string_roundtrip(self, store):
        plaintext = "x" * 10000
        encrypted = store.encrypt(plaintext)
        decrypted = store.decrypt(encrypted)
        assert decrypted == plaintext
