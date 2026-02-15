"""
Encryption Layer — Fernet Encryption at Rest for OpenClaw Empire Data
=====================================================================

Provides transparent encryption/decryption for all sensitive data files
under data/ (accounts, identities, auth, revenue, etc.) using Fernet
symmetric encryption (AES-128-CBC with HMAC-SHA256).

Features:
    - Key generation, rotation, revocation, and expiration tracking
    - Master key derived from OPENCLAW_MASTER_KEY env var (or auto-generated)
    - Encrypted key store (keys.json.enc) protects individual data keys
    - File-level encryption with header format: OPENCLAW_ENC_V1|{key_id}|{algo}
    - Atomic encrypted JSON read/write
    - Directory-level batch encrypt/decrypt with recursion
    - Secure deletion (random-byte overwrite before unlink)
    - SecureConfig: encrypted key-value store for sensitive settings
    - Full CLI with subcommands for all operations

Data stored under: data/encryption/
    keys.json.enc           -- encrypted key store (all Fernet keys)
    secure_config.enc       -- encrypted config key-value store
    .master_key             -- auto-generated master key (dev/insecure fallback)
    rotation_log.json       -- audit trail of key rotations

Usage:
    from src.encryption_layer import get_encryption_layer, encrypt, decrypt

    enc = get_encryption_layer()
    ciphertext = enc.encrypt_data({"api_key": "sk-secret-123"})
    plaintext  = enc.decrypt_data(ciphertext)

    enc.encrypt_file(Path("data/accounts/credentials.json"))
    enc.decrypt_file(Path("data/accounts/credentials.json.enc"))

    from src.encryption_layer import get_secure_config
    cfg = get_secure_config()
    cfg.set("openai_api_key", "sk-...")
    key = cfg.get("openai_api_key")

CLI:
    python -m src.encryption_layer init
    python -m src.encryption_layer encrypt-file --path data/accounts/credentials.json
    python -m src.encryption_layer decrypt-file --path data/accounts/credentials.json.enc
    python -m src.encryption_layer encrypt-dir --path data/accounts --recursive
    python -m src.encryption_layer decrypt-dir --path data/accounts --recursive
    python -m src.encryption_layer rotate
    python -m src.encryption_layer keys
    python -m src.encryption_layer status
    python -m src.encryption_layer secure-config get --key openai_api_key
    python -m src.encryption_layer secure-config set --key openai_api_key --value sk-...
    python -m src.encryption_layer secure-config list
    python -m src.encryption_layer secure-config delete --key openai_api_key
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import secrets
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("encryption_layer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
ENCRYPTION_DATA_DIR = BASE_DIR / "data" / "encryption"
KEYS_FILE = ENCRYPTION_DATA_DIR / "keys.json.enc"
SECURE_CONFIG_FILE = ENCRYPTION_DATA_DIR / "secure_config.enc"
MASTER_KEY_FILE = ENCRYPTION_DATA_DIR / ".master_key"
ROTATION_LOG_FILE = ENCRYPTION_DATA_DIR / "rotation_log.json"

ENCRYPTION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# File header prefix for encrypted files
HEADER_PREFIX = "OPENCLAW_ENC_V1"
HEADER_SEPARATOR = "|"

# Directories containing sensitive data (relative to BASE_DIR/data/)
DEFAULT_ENCRYPTED_DIRS = ["accounts", "auth", "revenue", "identities"]

# Secure deletion: number of overwrite passes
SECURE_DELETE_PASSES = 3

# Key expiration default (days) -- 0 means no expiration
DEFAULT_KEY_EXPIRY_DAYS = 365

# Maximum keys to retain (including rotated)
MAX_KEY_HISTORY = 50

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now_utc().isoformat()


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EncryptionAlgorithm(str, Enum):
    FERNET = "fernet"       # Default -- symmetric AES-128-CBC with HMAC-SHA256
    AES_GCM = "aes_gcm"    # Future: AES-256-GCM (not yet implemented)


class KeyStatus(str, Enum):
    ACTIVE = "active"
    ROTATED = "rotated"
    REVOKED = "revoked"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeyInfo:
    key_id: str                             # UUID
    algorithm: EncryptionAlgorithm
    created_at: str                         # ISO 8601
    rotated_at: Optional[str]               # ISO 8601 or None
    expires_at: Optional[str]               # ISO 8601 or None
    status: KeyStatus
    version: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["algorithm"] = self.algorithm.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeyInfo:
        return cls(
            key_id=data["key_id"],
            algorithm=EncryptionAlgorithm(data.get("algorithm", "fernet")),
            created_at=data["created_at"],
            rotated_at=data.get("rotated_at"),
            expires_at=data.get("expires_at"),
            status=KeyStatus(data.get("status", "active")),
            version=data.get("version", 1),
            description=data.get("description", ""),
        )

    @property
    def is_usable(self) -> bool:
        """Key can be used for decryption (active or rotated, not revoked/expired)."""
        return self.status in (KeyStatus.ACTIVE, KeyStatus.ROTATED)

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return _now_utc() > exp
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# KeyManager
# ---------------------------------------------------------------------------

class KeyManager:
    """Manages encryption keys with rotation support.

    The key store itself is encrypted with a master key derived from the
    OPENCLAW_MASTER_KEY environment variable.  If the env var is not set,
    a key is auto-generated and saved to data/encryption/.master_key
    (with a logged warning -- insecure for production).
    """

    def __init__(self) -> None:
        self._keys_file: Path = KEYS_FILE
        self._active_key_id: Optional[str] = None
        self._keys: Dict[str, Dict[str, Any]] = {}      # key_id -> {info, key_b64}
        self._master_fernet = None
        self._initialized = False

    # -- Initialization -----------------------------------------------------

    def initialize(self) -> None:
        """Load or create the key store."""
        if self._initialized:
            return
        self._ensure_master_key()
        self._load_key_store()
        self._check_expirations()
        self._initialized = True
        logger.debug("KeyManager initialized, %d keys loaded", len(self._keys))

    def _ensure_master_key(self) -> None:
        """Derive master Fernet key from env var or auto-generated file."""
        from cryptography.fernet import Fernet

        master_bytes = self._derive_master_key()
        # Fernet requires a url-safe base64-encoded 32-byte key
        # We use HKDF-like derivation: SHA-256 of master material, then base64
        derived = hashlib.sha256(master_bytes).digest()
        fernet_key = base64.urlsafe_b64encode(derived)
        self._master_fernet = Fernet(fernet_key)

    def _derive_master_key(self) -> bytes:
        """Get master key bytes from OPENCLAW_MASTER_KEY env or auto-generate."""
        env_key = os.environ.get("OPENCLAW_MASTER_KEY")
        if env_key:
            logger.debug("Using OPENCLAW_MASTER_KEY from environment")
            return env_key.encode("utf-8")

        # Fallback: auto-generated file
        if MASTER_KEY_FILE.exists():
            logger.debug("Using auto-generated master key from %s", MASTER_KEY_FILE)
            return MASTER_KEY_FILE.read_bytes()

        # Generate new master key
        logger.warning(
            "OPENCLAW_MASTER_KEY not set. Auto-generating master key to %s. "
            "This is INSECURE for production -- set the env var instead.",
            MASTER_KEY_FILE,
        )
        MASTER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        key_material = secrets.token_bytes(64)
        MASTER_KEY_FILE.write_bytes(key_material)
        # Restrict permissions where possible (best-effort on Windows)
        try:
            os.chmod(str(MASTER_KEY_FILE), 0o600)
        except (OSError, NotImplementedError):
            pass
        return key_material

    # -- Key store persistence ----------------------------------------------

    def _encrypt_key_store(self, data: Dict[str, Any]) -> bytes:
        """Encrypt the key store dict to bytes."""
        plaintext = json.dumps(data, indent=2, default=str).encode("utf-8")
        return self._master_fernet.encrypt(plaintext)

    def _decrypt_key_store(self, ciphertext: bytes) -> Dict[str, Any]:
        """Decrypt the key store bytes back to a dict."""
        plaintext = self._master_fernet.decrypt(ciphertext)
        return json.loads(plaintext.decode("utf-8"))

    def _load_key_store(self) -> None:
        """Load the encrypted key store from disk."""
        if not self._keys_file.exists():
            logger.debug("No key store found, starting fresh")
            self._keys = {}
            self._active_key_id = None
            return
        try:
            raw = self._keys_file.read_bytes()
            store = self._decrypt_key_store(raw)
            self._keys = store.get("keys", {})
            self._active_key_id = store.get("active_key_id")
            logger.debug(
                "Loaded key store: %d keys, active=%s",
                len(self._keys), self._active_key_id,
            )
        except Exception as exc:
            logger.error("Failed to decrypt key store: %s", exc)
            raise RuntimeError(
                "Cannot decrypt key store. Check OPENCLAW_MASTER_KEY or "
                "data/encryption/.master_key."
            ) from exc

    def _save_key_store(self) -> None:
        """Persist the encrypted key store to disk."""
        store = {
            "active_key_id": self._active_key_id,
            "keys": self._keys,
            "updated_at": _now_iso(),
        }
        encrypted = self._encrypt_key_store(store)
        self._keys_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._keys_file.with_suffix(".tmp")
        tmp.write_bytes(encrypted)
        os.replace(str(tmp), str(self._keys_file))
        logger.debug("Key store saved (%d bytes)", len(encrypted))

    # -- Key operations -----------------------------------------------------

    def generate_key(
        self,
        description: str = "",
        expires_days: int = DEFAULT_KEY_EXPIRY_DAYS,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
    ) -> KeyInfo:
        """Generate a new Fernet key and set it as the active key."""
        self.initialize()

        if algorithm != EncryptionAlgorithm.FERNET:
            raise NotImplementedError(
                f"Algorithm {algorithm.value} is not yet implemented. Use FERNET."
            )

        from cryptography.fernet import Fernet

        key_id = str(uuid.uuid4())
        key_bytes = Fernet.generate_key()  # url-safe base64 of 32 random bytes
        now = _now_iso()

        expires_at = None
        if expires_days and expires_days > 0:
            expires_at = (_now_utc() + timedelta(days=expires_days)).isoformat()

        # Determine version (max existing version + 1)
        max_version = 0
        for k in self._keys.values():
            v = k.get("info", {}).get("version", 0)
            if v > max_version:
                max_version = v

        info = KeyInfo(
            key_id=key_id,
            algorithm=algorithm,
            created_at=now,
            rotated_at=None,
            expires_at=expires_at,
            status=KeyStatus.ACTIVE,
            version=max_version + 1,
            description=description or f"Key v{max_version + 1}",
        )

        # If there was a previous active key, mark it as rotated
        if self._active_key_id and self._active_key_id in self._keys:
            prev = self._keys[self._active_key_id]
            prev["info"]["status"] = KeyStatus.ROTATED.value
            prev["info"]["rotated_at"] = now

        self._keys[key_id] = {
            "info": info.to_dict(),
            "key_b64": key_bytes.decode("ascii"),
        }
        self._active_key_id = key_id

        # Trim old keys beyond MAX_KEY_HISTORY
        self._trim_old_keys()

        self._save_key_store()
        logger.info("Generated key %s (v%d) -- %s", key_id[:8], info.version, info.description)
        return info

    def get_active_key(self) -> Tuple[str, bytes]:
        """Return (key_id, key_bytes) for the current active key.

        If no active key exists, generates one automatically.
        """
        self.initialize()

        if not self._active_key_id or self._active_key_id not in self._keys:
            logger.info("No active key found, generating initial key")
            info = self.generate_key(description="Auto-generated initial key")
            return info.key_id, self._get_key_bytes(info.key_id)

        entry = self._keys[self._active_key_id]
        info_dict = entry["info"]

        # Check expiration
        if info_dict.get("expires_at"):
            try:
                exp = datetime.fromisoformat(info_dict["expires_at"])
                if _now_utc() > exp:
                    logger.warning(
                        "Active key %s has expired, generating new key",
                        self._active_key_id[:8],
                    )
                    info_dict["status"] = KeyStatus.EXPIRED.value
                    new_info = self.generate_key(description="Auto-rotated (expired)")
                    return new_info.key_id, self._get_key_bytes(new_info.key_id)
            except (ValueError, TypeError):
                pass

        return self._active_key_id, self._get_key_bytes(self._active_key_id)

    def rotate_key(self, description: str = "") -> KeyInfo:
        """Create a new key and mark the current active key as rotated.

        Returns the new KeyInfo.
        """
        self.initialize()
        desc = description or f"Rotated from {self._active_key_id[:8] if self._active_key_id else 'none'}"
        new_info = self.generate_key(description=desc)

        # Log the rotation
        self._log_rotation(new_info.key_id)

        logger.info("Key rotated: new active key is %s (v%d)", new_info.key_id[:8], new_info.version)
        return new_info

    def revoke_key(self, key_id: str) -> bool:
        """Mark a key as revoked.  Cannot revoke the active key (rotate first)."""
        self.initialize()

        if key_id not in self._keys:
            logger.error("Key %s not found", key_id[:8])
            return False

        if key_id == self._active_key_id:
            logger.error("Cannot revoke the active key. Rotate first.")
            return False

        self._keys[key_id]["info"]["status"] = KeyStatus.REVOKED.value
        self._save_key_store()
        logger.info("Key %s revoked", key_id[:8])
        return True

    def get_key_by_id(self, key_id: str) -> bytes:
        """Retrieve key bytes for a specific key_id (for decryption of old files)."""
        self.initialize()
        return self._get_key_bytes(key_id)

    def _get_key_bytes(self, key_id: str) -> bytes:
        """Internal: extract raw key bytes from the store."""
        if key_id not in self._keys:
            raise KeyError(f"Key {key_id[:8]}... not found in key store")
        entry = self._keys[key_id]
        info = entry["info"]
        if info.get("status") == KeyStatus.REVOKED.value:
            raise ValueError(f"Key {key_id[:8]}... has been revoked and cannot be used")
        return entry["key_b64"].encode("ascii")

    def list_keys(self) -> List[KeyInfo]:
        """Return all keys as KeyInfo objects, sorted by version descending."""
        self.initialize()
        infos = []
        for entry in self._keys.values():
            infos.append(KeyInfo.from_dict(entry["info"]))
        infos.sort(key=lambda k: k.version, reverse=True)
        return infos

    def _check_expirations(self) -> None:
        """Mark any expired keys."""
        changed = False
        for key_id, entry in self._keys.items():
            info = entry["info"]
            if info.get("status") in (KeyStatus.REVOKED.value, KeyStatus.EXPIRED.value):
                continue
            expires_at = info.get("expires_at")
            if not expires_at:
                continue
            try:
                exp = datetime.fromisoformat(expires_at)
                if _now_utc() > exp:
                    info["status"] = KeyStatus.EXPIRED.value
                    changed = True
                    logger.warning("Key %s has expired", key_id[:8])
            except (ValueError, TypeError):
                pass
        if changed:
            self._save_key_store()

    def _trim_old_keys(self) -> None:
        """Remove oldest revoked/expired keys when count exceeds MAX_KEY_HISTORY."""
        if len(self._keys) <= MAX_KEY_HISTORY:
            return
        # Sort by version ascending, remove oldest non-active first
        sorted_ids = sorted(
            self._keys.keys(),
            key=lambda kid: self._keys[kid]["info"].get("version", 0),
        )
        removed = 0
        for kid in sorted_ids:
            if len(self._keys) - removed <= MAX_KEY_HISTORY:
                break
            if kid == self._active_key_id:
                continue
            status = self._keys[kid]["info"].get("status", "")
            if status in (KeyStatus.REVOKED.value, KeyStatus.EXPIRED.value):
                del self._keys[kid]
                removed += 1
                logger.debug("Trimmed old key %s", kid[:8])

    def _log_rotation(self, new_key_id: str) -> None:
        """Append to the rotation audit log."""
        log = _load_json(ROTATION_LOG_FILE, default=[])
        log.append({
            "timestamp": _now_iso(),
            "new_key_id": new_key_id,
            "old_key_id": self._active_key_id,
            "total_keys": len(self._keys),
        })
        # Keep last 200 entries
        if len(log) > 200:
            log = log[-200:]
        _save_json(ROTATION_LOG_FILE, log)

    def has_keys(self) -> bool:
        """Return True if any keys exist in the store."""
        self.initialize()
        return len(self._keys) > 0


# ---------------------------------------------------------------------------
# EncryptionLayer
# ---------------------------------------------------------------------------

class EncryptionLayer:
    """Encrypts/decrypts data files at rest using Fernet.

    File format:
        Line 1: OPENCLAW_ENC_V1|{key_id}|{algorithm}
        Line 2+: base64-encoded Fernet ciphertext

    Supports:
        - In-memory data encryption (str, bytes, dict -> base64 string)
        - File-level encryption (.enc extension added)
        - JSON-specific atomic encrypted read/write
        - Directory batch operations with recursion
        - Key rotation (re-encrypt all files with new key)
        - Encryption detection via header check
    """

    def __init__(self, key_manager: Optional[KeyManager] = None) -> None:
        self._key_manager = key_manager or KeyManager()
        self._encrypted_dirs: List[str] = list(DEFAULT_ENCRYPTED_DIRS)

    @property
    def key_manager(self) -> KeyManager:
        return self._key_manager

    # -- Data encryption (in-memory) ----------------------------------------

    async def encrypt_data(self, plaintext: Union[str, bytes, Dict]) -> str:
        """Encrypt data to a base64 string with embedded key reference.

        Args:
            plaintext: string, bytes, or dict (auto-serialized to JSON).

        Returns:
            A string in format: OPENCLAW_ENC_V1|{key_id}|fernet\\n{base64_ciphertext}
        """
        from cryptography.fernet import Fernet

        key_id, key_bytes = self._key_manager.get_active_key()
        f = Fernet(key_bytes)

        if isinstance(plaintext, dict):
            raw = json.dumps(plaintext, default=str).encode("utf-8")
        elif isinstance(plaintext, str):
            raw = plaintext.encode("utf-8")
        else:
            raw = plaintext

        token = f.encrypt(raw)
        header = f"{HEADER_PREFIX}{HEADER_SEPARATOR}{key_id}{HEADER_SEPARATOR}{EncryptionAlgorithm.FERNET.value}"
        return f"{header}\n{token.decode('ascii')}"

    def encrypt_data_sync(self, plaintext: Union[str, bytes, Dict]) -> str:
        return _run_sync(self.encrypt_data(plaintext))

    async def decrypt_data(self, ciphertext: str) -> Union[str, Dict]:
        """Decrypt a ciphertext string produced by encrypt_data().

        Automatically detects JSON content and returns a dict when applicable.
        """
        from cryptography.fernet import Fernet

        lines = ciphertext.strip().split("\n", 1)
        if len(lines) != 2:
            raise ValueError("Invalid encrypted data format: missing header or body")

        header_line, body = lines
        parts = header_line.split(HEADER_SEPARATOR)
        if len(parts) < 3 or parts[0] != HEADER_PREFIX:
            raise ValueError(f"Invalid encryption header: {header_line[:50]}")

        key_id = parts[1]
        algorithm = parts[2]

        if algorithm != EncryptionAlgorithm.FERNET.value:
            raise NotImplementedError(f"Unsupported algorithm: {algorithm}")

        key_bytes = self._key_manager.get_key_by_id(key_id)
        f = Fernet(key_bytes)
        decrypted = f.decrypt(body.encode("ascii"))
        text = decrypted.decode("utf-8")

        # Try to parse as JSON
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    def decrypt_data_sync(self, ciphertext: str) -> Union[str, Dict]:
        return _run_sync(self.decrypt_data(ciphertext))

    # -- File encryption ----------------------------------------------------

    async def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt a file in-place, adding .enc extension.

        The original file is securely deleted after successful encryption.
        Returns the path to the encrypted file.
        """
        from cryptography.fernet import Fernet

        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix == ".enc":
            logger.warning("File already has .enc extension, skipping: %s", file_path.name)
            return file_path

        if self.is_encrypted(file_path):
            logger.warning("File appears already encrypted, skipping: %s", file_path.name)
            return file_path

        key_id, key_bytes = self._key_manager.get_active_key()
        f = Fernet(key_bytes)

        # Read original
        raw = file_path.read_bytes()
        token = f.encrypt(raw)

        # Build header
        header = f"{HEADER_PREFIX}{HEADER_SEPARATOR}{key_id}{HEADER_SEPARATOR}{EncryptionAlgorithm.FERNET.value}\n"

        # Write encrypted file
        enc_path = file_path.with_suffix(file_path.suffix + ".enc")
        tmp_path = enc_path.with_suffix(".tmp")

        with open(tmp_path, "wb") as fh:
            fh.write(header.encode("utf-8"))
            fh.write(token)

        os.replace(str(tmp_path), str(enc_path))

        # Securely delete original
        self._secure_delete(file_path)

        logger.info("Encrypted: %s -> %s", file_path.name, enc_path.name)
        return enc_path

    def encrypt_file_sync(self, file_path: Path) -> Path:
        return _run_sync(self.encrypt_file(file_path))

    async def decrypt_file(self, file_path: Path) -> Path:
        """Decrypt a .enc file, restoring the original.

        The .enc file is securely deleted after successful decryption.
        Returns the path to the decrypted file.
        """
        from cryptography.fernet import Fernet

        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the encrypted file
        raw = file_path.read_bytes()

        # Split header from body
        newline_idx = raw.find(b"\n")
        if newline_idx < 0:
            raise ValueError(f"Invalid encrypted file (no header): {file_path.name}")

        header_line = raw[:newline_idx].decode("utf-8")
        body = raw[newline_idx + 1:]

        parts = header_line.split(HEADER_SEPARATOR)
        if len(parts) < 3 or parts[0] != HEADER_PREFIX:
            raise ValueError(f"Invalid encryption header in {file_path.name}: {header_line[:60]}")

        key_id = parts[1]
        algorithm = parts[2]

        if algorithm != EncryptionAlgorithm.FERNET.value:
            raise NotImplementedError(f"Unsupported algorithm: {algorithm}")

        key_bytes = self._key_manager.get_key_by_id(key_id)
        f = Fernet(key_bytes)
        decrypted = f.decrypt(body)

        # Determine output path: remove .enc suffix
        if file_path.suffix == ".enc":
            out_path = file_path.with_suffix("")
        else:
            out_path = file_path.with_suffix(".dec")

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as fh:
            fh.write(decrypted)
        os.replace(str(tmp_path), str(out_path))

        # Securely delete encrypted file
        self._secure_delete(file_path)

        logger.info("Decrypted: %s -> %s", file_path.name, out_path.name)
        return out_path

    def decrypt_file_sync(self, file_path: Path) -> Path:
        return _run_sync(self.decrypt_file(file_path))

    # -- JSON-specific encryption -------------------------------------------

    async def encrypt_json(self, data: Dict, output_path: Path) -> None:
        """Atomically write an encrypted JSON file.

        The resulting file contains the encryption header followed by
        the Fernet-encrypted JSON payload.
        """
        from cryptography.fernet import Fernet

        output_path = Path(output_path).resolve()
        key_id, key_bytes = self._key_manager.get_active_key()
        f = Fernet(key_bytes)

        plaintext = json.dumps(data, indent=2, default=str).encode("utf-8")
        token = f.encrypt(plaintext)

        header = f"{HEADER_PREFIX}{HEADER_SEPARATOR}{key_id}{HEADER_SEPARATOR}{EncryptionAlgorithm.FERNET.value}\n"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            fh.write(header.encode("utf-8"))
            fh.write(token)
        os.replace(str(tmp), str(output_path))
        logger.debug("Encrypted JSON written: %s", output_path.name)

    def encrypt_json_sync(self, data: Dict, output_path: Path) -> None:
        return _run_sync(self.encrypt_json(data, output_path))

    async def decrypt_json(self, file_path: Path) -> Dict:
        """Read and decrypt an encrypted JSON file."""
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Encrypted JSON not found: {file_path}")

        from cryptography.fernet import Fernet

        raw = file_path.read_bytes()
        newline_idx = raw.find(b"\n")
        if newline_idx < 0:
            raise ValueError(f"Invalid encrypted file (no header): {file_path.name}")

        header_line = raw[:newline_idx].decode("utf-8")
        body = raw[newline_idx + 1:]

        parts = header_line.split(HEADER_SEPARATOR)
        if len(parts) < 3 or parts[0] != HEADER_PREFIX:
            raise ValueError(f"Invalid encryption header: {header_line[:60]}")

        key_id = parts[1]
        algorithm = parts[2]

        if algorithm != EncryptionAlgorithm.FERNET.value:
            raise NotImplementedError(f"Unsupported algorithm: {algorithm}")

        key_bytes = self._key_manager.get_key_by_id(key_id)
        f = Fernet(key_bytes)
        decrypted = f.decrypt(body)
        return json.loads(decrypted.decode("utf-8"))

    def decrypt_json_sync(self, file_path: Path) -> Dict:
        return _run_sync(self.decrypt_json(file_path))

    # -- Directory operations -----------------------------------------------

    async def encrypt_directory(self, dir_path: Path, recursive: bool = True) -> Dict[str, Any]:
        """Encrypt all files in a directory.

        Args:
            dir_path: Directory to encrypt.
            recursive: If True, process subdirectories too.

        Returns:
            Stats dict with counts of encrypted, skipped, errored files.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        stats = {"encrypted": 0, "skipped": 0, "errors": 0, "files": []}
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix == ".enc":
                stats["skipped"] += 1
                continue
            if file_path.suffix == ".tmp":
                continue
            if file_path.name.startswith("."):
                stats["skipped"] += 1
                continue
            # Skip gitkeep
            if file_path.name == ".gitkeep":
                stats["skipped"] += 1
                continue

            try:
                enc_path = await self.encrypt_file(file_path)
                stats["encrypted"] += 1
                stats["files"].append(str(enc_path))
            except Exception as exc:
                logger.error("Failed to encrypt %s: %s", file_path.name, exc)
                stats["errors"] += 1

        logger.info(
            "Directory encrypt complete: %d encrypted, %d skipped, %d errors",
            stats["encrypted"], stats["skipped"], stats["errors"],
        )
        return stats

    def encrypt_directory_sync(self, dir_path: Path, recursive: bool = True) -> Dict[str, Any]:
        return _run_sync(self.encrypt_directory(dir_path, recursive))

    async def decrypt_directory(self, dir_path: Path, recursive: bool = True) -> Dict[str, Any]:
        """Decrypt all .enc files in a directory.

        Args:
            dir_path: Directory to decrypt.
            recursive: If True, process subdirectories too.

        Returns:
            Stats dict with counts of decrypted, skipped, errored files.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        stats = {"decrypted": 0, "skipped": 0, "errors": 0, "files": []}
        pattern = "**/*.enc" if recursive else "*.enc"

        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix == ".tmp":
                continue

            try:
                dec_path = await self.decrypt_file(file_path)
                stats["decrypted"] += 1
                stats["files"].append(str(dec_path))
            except Exception as exc:
                logger.error("Failed to decrypt %s: %s", file_path.name, exc)
                stats["errors"] += 1

        logger.info(
            "Directory decrypt complete: %d decrypted, %d skipped, %d errors",
            stats["decrypted"], stats["skipped"], stats["errors"],
        )
        return stats

    def decrypt_directory_sync(self, dir_path: Path, recursive: bool = True) -> Dict[str, Any]:
        return _run_sync(self.decrypt_directory(dir_path, recursive))

    # -- Key rotation -------------------------------------------------------

    async def rotate_encryption(
        self,
        old_key_id: Optional[str] = None,
        new_key_id: Optional[str] = None,
        target_dirs: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Re-encrypt all files from old_key to new_key.

        If old_key_id is None, re-encrypts files encrypted with ANY non-active key.
        If new_key_id is None, uses the current active key (rotating first if needed).

        Args:
            old_key_id: Specific key to rotate from (None = all non-active).
            new_key_id: Specific key to rotate to (None = active key).
            target_dirs: Directories to scan (None = default encrypted dirs).

        Returns:
            Stats dict.
        """
        from cryptography.fernet import Fernet

        # Ensure we have a new key
        if new_key_id:
            new_key_bytes = self._key_manager.get_key_by_id(new_key_id)
        else:
            new_key_id, new_key_bytes = self._key_manager.get_active_key()

        new_fernet = Fernet(new_key_bytes)

        # Determine directories
        if target_dirs is None:
            data_dir = BASE_DIR / "data"
            target_dirs = [data_dir / d for d in self._encrypted_dirs if (data_dir / d).is_dir()]

        stats = {"rotated": 0, "skipped": 0, "errors": 0, "files": []}

        for dir_path in target_dirs:
            if not dir_path.is_dir():
                continue

            for file_path in sorted(dir_path.rglob("*.enc")):
                if not file_path.is_file():
                    continue

                try:
                    raw = file_path.read_bytes()
                    newline_idx = raw.find(b"\n")
                    if newline_idx < 0:
                        stats["skipped"] += 1
                        continue

                    header_line = raw[:newline_idx].decode("utf-8")
                    body = raw[newline_idx + 1:]

                    parts = header_line.split(HEADER_SEPARATOR)
                    if len(parts) < 3 or parts[0] != HEADER_PREFIX:
                        stats["skipped"] += 1
                        continue

                    file_key_id = parts[1]

                    # Skip if already using new key
                    if file_key_id == new_key_id:
                        stats["skipped"] += 1
                        continue

                    # Skip if old_key_id specified and doesn't match
                    if old_key_id and file_key_id != old_key_id:
                        stats["skipped"] += 1
                        continue

                    # Decrypt with old key
                    old_key_bytes = self._key_manager.get_key_by_id(file_key_id)
                    old_fernet = Fernet(old_key_bytes)
                    plaintext = old_fernet.decrypt(body)

                    # Re-encrypt with new key
                    new_token = new_fernet.encrypt(plaintext)
                    new_header = f"{HEADER_PREFIX}{HEADER_SEPARATOR}{new_key_id}{HEADER_SEPARATOR}{EncryptionAlgorithm.FERNET.value}\n"

                    tmp = file_path.with_suffix(".tmp")
                    with open(tmp, "wb") as fh:
                        fh.write(new_header.encode("utf-8"))
                        fh.write(new_token)
                    os.replace(str(tmp), str(file_path))

                    stats["rotated"] += 1
                    stats["files"].append(str(file_path))
                    logger.debug("Rotated: %s", file_path.name)

                except Exception as exc:
                    logger.error("Failed to rotate %s: %s", file_path.name, exc)
                    stats["errors"] += 1

        logger.info(
            "Key rotation complete: %d rotated, %d skipped, %d errors",
            stats["rotated"], stats["skipped"], stats["errors"],
        )
        return stats

    def rotate_encryption_sync(
        self,
        old_key_id: Optional[str] = None,
        new_key_id: Optional[str] = None,
        target_dirs: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        return _run_sync(self.rotate_encryption(old_key_id, new_key_id, target_dirs))

    # -- Detection & inspection ---------------------------------------------

    def is_encrypted(self, file_path: Path) -> bool:
        """Check if a file has the OpenClaw encryption header."""
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return False
        try:
            with open(file_path, "rb") as fh:
                first_line = fh.readline(512)
            header = first_line.decode("utf-8", errors="ignore").strip()
            parts = header.split(HEADER_SEPARATOR)
            return len(parts) >= 3 and parts[0] == HEADER_PREFIX
        except Exception:
            return False

    def get_file_key_id(self, file_path: Path) -> Optional[str]:
        """Extract the key_id from an encrypted file's header."""
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "rb") as fh:
                first_line = fh.readline(512)
            header = first_line.decode("utf-8", errors="ignore").strip()
            parts = header.split(HEADER_SEPARATOR)
            if len(parts) >= 3 and parts[0] == HEADER_PREFIX:
                return parts[1]
        except Exception:
            pass
        return None

    async def get_stats(self) -> Dict[str, Any]:
        """Gather encryption statistics across all data directories."""
        self._key_manager.initialize()
        data_dir = BASE_DIR / "data"

        total_encrypted = 0
        total_unencrypted = 0
        total_enc_size = 0
        total_plain_size = 0
        key_usage: Dict[str, int] = {}
        dir_stats: Dict[str, Dict[str, int]] = {}

        for subdir_name in self._encrypted_dirs:
            subdir = data_dir / subdir_name
            if not subdir.is_dir():
                continue

            enc_count = 0
            plain_count = 0
            enc_size = 0
            plain_size = 0

            for fp in subdir.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.name.startswith(".") or fp.suffix == ".tmp":
                    continue

                fsize = fp.stat().st_size
                if fp.suffix == ".enc" or self.is_encrypted(fp):
                    enc_count += 1
                    enc_size += fsize
                    kid = self.get_file_key_id(fp)
                    if kid:
                        key_usage[kid] = key_usage.get(kid, 0) + 1
                else:
                    plain_count += 1
                    plain_size += fsize

            total_encrypted += enc_count
            total_unencrypted += plain_count
            total_enc_size += enc_size
            total_plain_size += plain_size
            dir_stats[subdir_name] = {
                "encrypted": enc_count,
                "unencrypted": plain_count,
                "encrypted_bytes": enc_size,
                "unencrypted_bytes": plain_size,
            }

        keys = self._key_manager.list_keys()
        active = [k for k in keys if k.status == KeyStatus.ACTIVE]
        rotated = [k for k in keys if k.status == KeyStatus.ROTATED]

        rotation_log = _load_json(ROTATION_LOG_FILE, default=[])
        last_rotation = rotation_log[-1]["timestamp"] if rotation_log else None

        return {
            "total_encrypted_files": total_encrypted,
            "total_unencrypted_files": total_unencrypted,
            "total_encrypted_bytes": total_enc_size,
            "total_unencrypted_bytes": total_plain_size,
            "total_keys": len(keys),
            "active_keys": len(active),
            "rotated_keys": len(rotated),
            "key_usage": key_usage,
            "directory_stats": dir_stats,
            "last_rotation": last_rotation,
            "master_key_source": "env" if os.environ.get("OPENCLAW_MASTER_KEY") else "file",
            "encryption_algorithm": EncryptionAlgorithm.FERNET.value,
            "checked_at": _now_iso(),
        }

    def get_stats_sync(self) -> Dict[str, Any]:
        return _run_sync(self.get_stats())

    # -- Secure deletion ----------------------------------------------------

    @staticmethod
    def _secure_delete(path: Path) -> None:
        """Overwrite file with random bytes before deleting.

        Performs SECURE_DELETE_PASSES passes of random overwrites, then
        a final zeros pass, then unlinks.  Best-effort on copy-on-write
        filesystems (SSD/NTFS may retain data regardless).
        """
        path = Path(path)
        if not path.exists() or not path.is_file():
            return

        try:
            file_size = path.stat().st_size
            if file_size == 0:
                path.unlink()
                return

            with open(path, "r+b") as fh:
                for _pass in range(SECURE_DELETE_PASSES):
                    fh.seek(0)
                    fh.write(secrets.token_bytes(file_size))
                    fh.flush()
                    os.fsync(fh.fileno())
                # Final zeros pass
                fh.seek(0)
                fh.write(b"\x00" * file_size)
                fh.flush()
                os.fsync(fh.fileno())

            path.unlink()
            logger.debug("Securely deleted: %s", path.name)
        except PermissionError:
            # Fallback: simple delete
            logger.warning("Secure delete failed (permissions), using simple delete: %s", path.name)
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.error("Secure delete error for %s: %s", path.name, exc)
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SecureConfig
# ---------------------------------------------------------------------------

class SecureConfig:
    """Encrypted configuration manager for sensitive settings.

    Stores key-value pairs in an encrypted file.  Values can be any
    JSON-serializable type.  Keys are plain strings stored alongside
    their encrypted values.
    """

    def __init__(self, encryption: Optional[EncryptionLayer] = None) -> None:
        self._encryption = encryption or get_encryption_layer()
        self._config_file: Path = SECURE_CONFIG_FILE
        self._cache: Optional[Dict[str, Any]] = None

    def _load(self) -> Dict[str, Any]:
        """Load the config from the encrypted file."""
        if self._cache is not None:
            return self._cache

        if not self._config_file.exists():
            self._cache = {}
            return self._cache

        try:
            self._cache = self._encryption.decrypt_json_sync(self._config_file)
        except Exception as exc:
            logger.error("Failed to load secure config: %s", exc)
            self._cache = {}
        return self._cache

    def _save(self) -> None:
        """Persist the config to the encrypted file."""
        data = self._cache if self._cache is not None else {}
        data["_updated_at"] = _now_iso()
        self._encryption.encrypt_json_sync(data, self._config_file)
        logger.debug("Secure config saved (%d keys)", len(data) - 1)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a config value by key."""
        data = self._load()
        return data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value (encrypts and saves)."""
        if key.startswith("_"):
            raise ValueError("Config keys cannot start with underscore (reserved)")
        data = self._load()
        data[key] = value
        self._save()
        logger.info("Secure config key set: %s", key)

    def delete(self, key: str) -> bool:
        """Remove a config key.  Returns True if the key existed."""
        data = self._load()
        if key in data:
            del data[key]
            self._save()
            logger.info("Secure config key deleted: %s", key)
            return True
        return False

    def list_keys(self) -> List[str]:
        """List all config keys (excluding internal _ prefixed keys)."""
        data = self._load()
        return sorted(k for k in data if not k.startswith("_"))

    def export_redacted(self) -> Dict[str, str]:
        """Export all keys with redacted values for debugging.

        Values are replaced with "***" to avoid leaking secrets.
        """
        data = self._load()
        redacted = {}
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str) and len(value) > 6:
                redacted[key] = value[:3] + "***" + value[-3:]
            elif isinstance(value, str):
                redacted[key] = "***"
            elif isinstance(value, (int, float)):
                redacted[key] = "***"
            elif isinstance(value, bool):
                redacted[key] = "***"
            elif isinstance(value, dict):
                redacted[key] = "{***}"
            elif isinstance(value, list):
                redacted[key] = f"[*** ({len(value)} items)]"
            else:
                redacted[key] = "***"
        return redacted

    def has(self, key: str) -> bool:
        """Check if a key exists in the config."""
        data = self._load()
        return key in data

    def clear(self) -> None:
        """Remove all config entries."""
        self._cache = {}
        self._save()
        logger.info("Secure config cleared")

    def count(self) -> int:
        """Return the number of config entries (excluding internal keys)."""
        return len(self.list_keys())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_encryption_layer: Optional[EncryptionLayer] = None
_secure_config: Optional[SecureConfig] = None


def get_encryption_layer() -> EncryptionLayer:
    """Get the singleton EncryptionLayer instance."""
    global _encryption_layer
    if _encryption_layer is None:
        _encryption_layer = EncryptionLayer()
    return _encryption_layer


def get_secure_config() -> SecureConfig:
    """Get the singleton SecureConfig instance."""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfig()
    return _secure_config


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def encrypt(data: Union[str, Dict]) -> str:
    """Convenience: encrypt data using the global EncryptionLayer."""
    return get_encryption_layer().encrypt_data_sync(data)


def decrypt(ciphertext: str) -> Union[str, Dict]:
    """Convenience: decrypt data using the global EncryptionLayer."""
    return get_encryption_layer().decrypt_data_sync(ciphertext)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_init(args: argparse.Namespace) -> None:
    """Initialize encryption: generate master key and first data key."""
    layer = get_encryption_layer()
    km = layer.key_manager
    km.initialize()

    if km.has_keys():
        print("Encryption already initialized.")
        keys = km.list_keys()
        active = [k for k in keys if k.status == KeyStatus.ACTIVE]
        if active:
            print(f"Active key: {active[0].key_id[:8]}... (v{active[0].version})")
        print(f"Total keys: {len(keys)}")
    else:
        info = km.generate_key(
            description=args.description or "Initial key",
            expires_days=args.expires_days,
        )
        print(f"Encryption initialized.")
        print(f"Key ID:      {info.key_id[:8]}...")
        print(f"Version:     {info.version}")
        print(f"Algorithm:   {info.algorithm.value}")
        print(f"Expires:     {info.expires_at or 'never'}")

    # Report master key source
    if os.environ.get("OPENCLAW_MASTER_KEY"):
        print("Master key:  from OPENCLAW_MASTER_KEY env var")
    else:
        print(f"Master key:  from {MASTER_KEY_FILE}")
        print("WARNING:     Set OPENCLAW_MASTER_KEY env var for production use!")


def _cli_encrypt_file(args: argparse.Namespace) -> None:
    """Encrypt a single file."""
    file_path = Path(args.path).resolve()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    layer = get_encryption_layer()
    enc_path = layer.encrypt_file_sync(file_path)
    print(f"Encrypted: {enc_path}")


def _cli_decrypt_file(args: argparse.Namespace) -> None:
    """Decrypt a single file."""
    file_path = Path(args.path).resolve()
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    layer = get_encryption_layer()
    dec_path = layer.decrypt_file_sync(file_path)
    print(f"Decrypted: {dec_path}")


def _cli_encrypt_dir(args: argparse.Namespace) -> None:
    """Encrypt all files in a directory."""
    dir_path = Path(args.path).resolve()
    if not dir_path.is_dir():
        print(f"Error: Not a directory: {dir_path}")
        sys.exit(1)
    layer = get_encryption_layer()
    stats = layer.encrypt_directory_sync(dir_path, recursive=args.recursive)
    _print_json(stats)


def _cli_decrypt_dir(args: argparse.Namespace) -> None:
    """Decrypt all .enc files in a directory."""
    dir_path = Path(args.path).resolve()
    if not dir_path.is_dir():
        print(f"Error: Not a directory: {dir_path}")
        sys.exit(1)
    layer = get_encryption_layer()
    stats = layer.decrypt_directory_sync(dir_path, recursive=args.recursive)
    _print_json(stats)


def _cli_rotate(args: argparse.Namespace) -> None:
    """Rotate keys and re-encrypt all files."""
    layer = get_encryption_layer()
    km = layer.key_manager

    print("Generating new encryption key...")
    new_info = km.rotate_key(description=args.description or "Manual rotation via CLI")
    print(f"New key: {new_info.key_id[:8]}... (v{new_info.version})")

    if not args.skip_reencrypt:
        print("Re-encrypting files with new key...")
        stats = layer.rotate_encryption_sync()
        print(f"Rotated:  {stats['rotated']} files")
        print(f"Skipped:  {stats['skipped']} files")
        print(f"Errors:   {stats['errors']} files")
    else:
        print("Skipped re-encryption (--skip-reencrypt). Run manually later.")


def _cli_keys(args: argparse.Namespace) -> None:
    """List all encryption keys."""
    layer = get_encryption_layer()
    km = layer.key_manager
    keys = km.list_keys()

    if not keys:
        print("No keys found. Run 'init' first.")
        return

    print(f"{'ID':<12} {'Ver':>4} {'Status':<10} {'Algorithm':<10} {'Created':<26} {'Expires':<26} {'Description'}")
    print("-" * 120)
    for k in keys:
        kid_short = k.key_id[:10] + ".."
        expires = k.expires_at[:19] if k.expires_at else "never"
        created = k.created_at[:19] if k.created_at else "?"
        print(f"{kid_short:<12} {k.version:>4} {k.status.value:<10} {k.algorithm.value:<10} {created:<26} {expires:<26} {k.description}")

    active = [k for k in keys if k.status == KeyStatus.ACTIVE]
    if active:
        print(f"\nActive key: {active[0].key_id[:8]}... (v{active[0].version})")


def _cli_status(args: argparse.Namespace) -> None:
    """Show encryption status and statistics."""
    layer = get_encryption_layer()
    stats = layer.get_stats_sync()

    print("=== OpenClaw Encryption Status ===\n")
    print(f"Algorithm:          {stats['encryption_algorithm']}")
    print(f"Master key source:  {stats['master_key_source']}")
    print(f"Total keys:         {stats['total_keys']}")
    print(f"Active keys:        {stats['active_keys']}")
    print(f"Rotated keys:       {stats['rotated_keys']}")
    print(f"Last rotation:      {stats['last_rotation'] or 'never'}")

    print(f"\n--- File Statistics ---")
    print(f"Encrypted files:    {stats['total_encrypted_files']}")
    print(f"Unencrypted files:  {stats['total_unencrypted_files']}")
    print(f"Encrypted size:     {_format_bytes(stats['total_encrypted_bytes'])}")
    print(f"Unencrypted size:   {_format_bytes(stats['total_unencrypted_bytes'])}")

    if stats["directory_stats"]:
        print(f"\n--- Per-Directory ---")
        for dirname, ds in sorted(stats["directory_stats"].items()):
            total = ds["encrypted"] + ds["unencrypted"]
            pct = (ds["encrypted"] / total * 100) if total > 0 else 0
            print(f"  {dirname:<20} {ds['encrypted']}/{total} encrypted ({pct:.0f}%)")

    if stats["key_usage"]:
        print(f"\n--- Key Usage ---")
        for kid, count in sorted(stats["key_usage"].items(), key=lambda x: -x[1]):
            print(f"  {kid[:10]}..  {count} files")

    print(f"\nChecked at: {stats['checked_at']}")


def _cli_secure_config(args: argparse.Namespace) -> None:
    """Manage secure configuration values."""
    cfg = get_secure_config()
    action = args.action

    if action == "get":
        if not args.key:
            print("Error: --key is required for 'get'")
            sys.exit(1)
        value = cfg.get(args.key)
        if value is None:
            print(f"Key '{args.key}' not found")
            sys.exit(1)
        if isinstance(value, (dict, list)):
            _print_json(value)
        else:
            print(value)

    elif action == "set":
        if not args.key or args.value is None:
            print("Error: --key and --value are required for 'set'")
            sys.exit(1)
        # Try to parse value as JSON
        try:
            parsed = json.loads(args.value)
            cfg.set(args.key, parsed)
        except (json.JSONDecodeError, ValueError):
            cfg.set(args.key, args.value)
        print(f"Set: {args.key}")

    elif action == "list":
        keys = cfg.list_keys()
        if not keys:
            print("No config keys stored.")
            return
        if args.redacted:
            _print_json(cfg.export_redacted())
        else:
            for k in keys:
                print(f"  {k}")
            print(f"\n{len(keys)} key(s) stored")

    elif action == "delete":
        if not args.key:
            print("Error: --key is required for 'delete'")
            sys.exit(1)
        if cfg.delete(args.key):
            print(f"Deleted: {args.key}")
        else:
            print(f"Key '{args.key}' not found")

    elif action == "export":
        _print_json(cfg.export_redacted())

    else:
        print(f"Unknown secure-config action: {action}")
        sys.exit(1)


def _format_bytes(size: int) -> str:
    """Format byte count into human-readable string."""
    if size == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    fsize = float(size)
    while fsize >= 1024 and i < len(units) - 1:
        fsize /= 1024
        i += 1
    return f"{fsize:.1f} {units[i]}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="encryption_layer",
        description="OpenClaw Empire -- Fernet Encryption at Rest",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Initialize encryption (generate master + data key)")
    init_p.add_argument("--description", default="", help="Key description")
    init_p.add_argument("--expires-days", type=int, default=DEFAULT_KEY_EXPIRY_DAYS, help="Key expiry in days (0=never)")
    init_p.set_defaults(func=_cli_init)

    # encrypt-file
    ef = sub.add_parser("encrypt-file", help="Encrypt a single file")
    ef.add_argument("--path", required=True, help="File to encrypt")
    ef.set_defaults(func=_cli_encrypt_file)

    # decrypt-file
    df = sub.add_parser("decrypt-file", help="Decrypt a single file")
    df.add_argument("--path", required=True, help="File to decrypt")
    df.set_defaults(func=_cli_decrypt_file)

    # encrypt-dir
    ed = sub.add_parser("encrypt-dir", help="Encrypt all files in a directory")
    ed.add_argument("--path", required=True, help="Directory to encrypt")
    ed.add_argument("--recursive", action="store_true", default=True, help="Recurse into subdirs (default: True)")
    ed.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse")
    ed.set_defaults(func=_cli_encrypt_dir)

    # decrypt-dir
    dd = sub.add_parser("decrypt-dir", help="Decrypt all .enc files in a directory")
    dd.add_argument("--path", required=True, help="Directory to decrypt")
    dd.add_argument("--recursive", action="store_true", default=True, help="Recurse into subdirs (default: True)")
    dd.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not recurse")
    dd.set_defaults(func=_cli_decrypt_dir)

    # rotate
    rot = sub.add_parser("rotate", help="Rotate keys and re-encrypt all files")
    rot.add_argument("--description", default="", help="Description for new key")
    rot.add_argument("--skip-reencrypt", action="store_true", help="Only create new key, skip file re-encryption")
    rot.set_defaults(func=_cli_rotate)

    # keys
    keys_p = sub.add_parser("keys", help="List all encryption keys")
    keys_p.set_defaults(func=_cli_keys)

    # status
    stat_p = sub.add_parser("status", help="Show encryption status and statistics")
    stat_p.set_defaults(func=_cli_status)

    # secure-config
    sc = sub.add_parser("secure-config", help="Manage encrypted config values")
    sc.add_argument("action", choices=["get", "set", "list", "delete", "export"], help="Config action")
    sc.add_argument("--key", default=None, help="Config key name")
    sc.add_argument("--value", default=None, help="Config value (for 'set')")
    sc.add_argument("--redacted", action="store_true", help="Show redacted values (for 'list')")
    sc.set_defaults(func=_cli_secure_config)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
