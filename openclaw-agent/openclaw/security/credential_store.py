"""Fernet-encrypted credential storage for platform passwords and tokens."""

from __future__ import annotations

import json
import os
from pathlib import Path

from cryptography.fernet import Fernet


class CredentialStore:
    """Encrypt and decrypt credentials using Fernet symmetric encryption.

    Key is loaded from OPENCLAW_ENCRYPTION_KEY env var or auto-generated
    and saved to data/encryption.key on first use.
    """

    def __init__(self, key_path: str | None = None):
        self._key_path = Path(key_path or self._default_key_path())
        self._fernet: Fernet | None = None

    @staticmethod
    def _default_key_path() -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "encryption.key",
        )

    def _load_or_create_key(self) -> bytes:
        env_key = os.environ.get("OPENCLAW_ENCRYPTION_KEY")
        if env_key:
            return env_key.encode()

        if self._key_path.exists():
            return self._key_path.read_bytes().strip()

        key = Fernet.generate_key()
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path.write_bytes(key)
        return key

    @property
    def fernet(self) -> Fernet:
        if self._fernet is None:
            self._fernet = Fernet(self._load_or_create_key())
        return self._fernet

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        return self.fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext back to plaintext."""
        return self.fernet.decrypt(ciphertext.encode()).decode()

    def encrypt_dict(self, data: dict) -> str:
        """Encrypt a dict as JSON."""
        return self.encrypt(json.dumps(data))

    def decrypt_dict(self, ciphertext: str) -> dict:
        """Decrypt a JSON dict."""
        return json.loads(self.decrypt(ciphertext))

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet key (for initial setup)."""
        return Fernet.generate_key().decode()
