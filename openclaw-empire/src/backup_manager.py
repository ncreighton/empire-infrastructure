"""
Backup Manager — State Backup/Restore for OpenClaw Empire Data
==============================================================

Full backup and restore system for all data/ directories across
Nick Creighton's 16-site WordPress publishing empire.  Supports full,
incremental, selective, and snapshot backup types with ZIP and gzip/tar
compression, SHA-256 integrity verification, and automatic pruning.

Backup Targets (data/ subdirectories):
    forge, amplify, vision, screenshots, scheduler, revenue, social,
    calendar, auth, content, accounts, memory, audit, circuit_breaker,
    encryption, performance, prompts, rate_limits, quality, rag, backups,
    identities

Features:
    - Full backup: all data/ directories in a single archive
    - Incremental backup: only files changed since last backup
    - Selective backup: specific directories only
    - Snapshot: metadata + file listing (no data copied)
    - ZIP and gzip/tar compression
    - SHA-256 checksum verification
    - Dry-run restore (preview without writing)
    - Selective restore from full backups
    - Disk usage analytics and diff-with-current
    - Automatic pruning of oldest backups
    - Async core with synchronous wrappers
    - Singleton access via get_backup_manager()
    - CLI with subcommands for all operations

All data stored under: data/backups/

Usage:
    from src.backup_manager import get_backup_manager, BackupType, CompressionType

    bm = get_backup_manager()

    # Create full backup
    manifest = bm.create_full_backup_sync(description="Pre-deploy safety net")

    # Create incremental backup (only changed files)
    manifest = bm.create_incremental_backup_sync(description="Nightly delta")

    # Create selective backup (specific directories)
    manifest = bm.create_selective_backup_sync(
        directories=["accounts", "encryption", "auth"],
        description="Sensitive data backup",
    )

    # Quick metadata snapshot
    manifest = bm.create_snapshot_sync(description="State checkpoint")

    # Restore a backup (dry run first)
    report = bm.restore_sync(manifest.backup_id, dry_run=True)
    report = bm.restore_sync(manifest.backup_id)

    # Verify integrity
    result = bm.verify_sync(manifest.backup_id)

    # Analytics
    stats = bm.get_stats()
    usage = bm.get_disk_usage()
    diff  = bm.diff_with_current(manifest.backup_id)

    # Prune old backups
    deleted = bm.prune_old(keep=30)

CLI:
    python -m src.backup_manager create --type full --description "Before deploy"
    python -m src.backup_manager create --type incremental --description "Nightly"
    python -m src.backup_manager create --type selective --dirs accounts auth encryption
    python -m src.backup_manager snapshot --description "State checkpoint"
    python -m src.backup_manager restore --id BACKUP_ID --dry-run
    python -m src.backup_manager restore --id BACKUP_ID --dirs accounts auth
    python -m src.backup_manager verify --id BACKUP_ID
    python -m src.backup_manager list --type full --limit 10
    python -m src.backup_manager delete --id BACKUP_ID
    python -m src.backup_manager prune --keep 20
    python -m src.backup_manager stats
    python -m src.backup_manager diff --id BACKUP_ID
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("backup_manager")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
DATA_DIR = BASE_DIR / "data"
BACKUP_DATA_DIR = BASE_DIR / "data" / "backups"
MANIFESTS_FILE = BACKUP_DATA_DIR / "manifests.json"

# Ensure data directory exists on import
BACKUP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MANIFESTS = 100  # bounded in-memory manifest list
DEFAULT_MAX_BACKUPS = 30  # auto-prune threshold
DEFAULT_KEEP_BACKUPS = 30
HASH_BUFFER_SIZE = 65536  # 64 KB read chunks for SHA-256
ARCHIVE_SUBDIR = "archives"  # subdirectory under BACKUP_DATA_DIR for archives
SNAPSHOT_SUBDIR = "snapshots"  # subdirectory for snapshot JSON files

# All data/ subdirectories eligible for backup
BACKUP_TARGETS = [
    "forge",
    "amplify",
    "vision",
    "screenshots",
    "scheduler",
    "revenue",
    "social",
    "calendar",
    "auth",
    "content",
    "accounts",
    "memory",
    "audit",
    "circuit_breaker",
    "encryption",
    "performance",
    "prompts",
    "rate_limits",
    "quality",
    "rag",
    "backups",
    "identities",
]


# ---------------------------------------------------------------------------
# Helpers — Time
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    if s is None or s == "":
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Helpers — Atomic JSON
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
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        # Atomic replace
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Helpers — Async/sync dual interface
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine from synchronous code, handling nested loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers — Formatting
# ---------------------------------------------------------------------------

def _human_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string (e.g. '4.2 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _short_id(backup_id: str) -> str:
    """Return the first 8 characters of a backup UUID for display."""
    return backup_id[:8] if len(backup_id) >= 8 else backup_id


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BackupType(str, Enum):
    """Type of backup operation."""
    FULL = "full"                # All data/ directories
    INCREMENTAL = "incremental"  # Only changed files since last backup
    SELECTIVE = "selective"      # Specific directories only
    SNAPSHOT = "snapshot"        # Quick state snapshot (metadata only)


class BackupStatus(str, Enum):
    """Current status of a backup."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RESTORING = "restoring"
    VERIFIED = "verified"


class CompressionType(str, Enum):
    """Compression algorithm for backup archives."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"


# ---------------------------------------------------------------------------
# Data class — BackupManifest
# ---------------------------------------------------------------------------

@dataclass
class BackupManifest:
    """Describes a single backup: what was backed up, where, when, how."""

    backup_id: str                          # UUID-4
    backup_type: BackupType
    status: BackupStatus
    created_at: str                         # ISO-8601
    completed_at: Optional[str]             # ISO-8601 or None
    source_dir: str                         # absolute path to data/
    backup_path: str                        # absolute path to archive or snapshot
    file_count: int                         # number of files included
    total_size_bytes: int                   # uncompressed total size
    compressed_size_bytes: int              # compressed archive size (0 for snapshots)
    compression: CompressionType
    directories_included: List[str]         # which data/ subdirs
    description: str
    checksum: Optional[str]                 # SHA-256 of the archive file
    parent_backup_id: Optional[str]         # for incremental: the base backup
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dictionary."""
        d = asdict(self)
        d["backup_type"] = self.backup_type.value
        d["status"] = self.status.value
        d["compression"] = self.compression.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BackupManifest:
        """Reconstruct from a dictionary (loaded from JSON)."""
        return cls(
            backup_id=d["backup_id"],
            backup_type=BackupType(d["backup_type"]),
            status=BackupStatus(d["status"]),
            created_at=d["created_at"],
            completed_at=d.get("completed_at"),
            source_dir=d["source_dir"],
            backup_path=d.get("backup_path", ""),
            file_count=d.get("file_count", 0),
            total_size_bytes=d.get("total_size_bytes", 0),
            compressed_size_bytes=d.get("compressed_size_bytes", 0),
            compression=CompressionType(d.get("compression", "zip")),
            directories_included=d.get("directories_included", []),
            description=d.get("description", ""),
            checksum=d.get("checksum"),
            parent_backup_id=d.get("parent_backup_id"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# BackupManager
# ---------------------------------------------------------------------------

class BackupManager:
    """State backup and restore manager for all data/ directories.

    Manages full, incremental, selective, and snapshot backups with ZIP or
    gzip/tar compression.  Provides restore, verification, pruning, disk
    usage analytics, and diff-with-current capabilities.

    Access via the module-level singleton ``get_backup_manager()``.
    """

    # Class-level defaults
    BACKUP_TARGETS = BACKUP_TARGETS

    def __init__(
        self,
        max_backups: int = DEFAULT_MAX_BACKUPS,
        default_compression: CompressionType = CompressionType.ZIP,
    ) -> None:
        self._max_backups = max_backups
        self._default_compression = default_compression
        self._manifests: List[BackupManifest] = []
        self._archives_dir = BACKUP_DATA_DIR / ARCHIVE_SUBDIR
        self._snapshots_dir = BACKUP_DATA_DIR / SNAPSHOT_SUBDIR
        self._archives_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._load_manifests()
        logger.info(
            "BackupManager initialized — %d existing backups, max=%d, compression=%s",
            len(self._manifests),
            self._max_backups,
            self._default_compression.value,
        )

    # -----------------------------------------------------------------------
    # Manifest persistence
    # -----------------------------------------------------------------------

    def _load_manifests(self) -> None:
        """Load manifest list from disk."""
        raw = _load_json(MANIFESTS_FILE, default=[])
        if not isinstance(raw, list):
            raw = []
        self._manifests = []
        for entry in raw:
            try:
                self._manifests.append(BackupManifest.from_dict(entry))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping corrupt manifest entry: %s", exc)
        # Enforce bound
        if len(self._manifests) > MAX_MANIFESTS:
            self._manifests = self._manifests[-MAX_MANIFESTS:]
        logger.debug("Loaded %d manifests from disk", len(self._manifests))

    def _save_manifests(self) -> None:
        """Persist the manifest list to disk atomically."""
        # Enforce bound before saving
        if len(self._manifests) > MAX_MANIFESTS:
            self._manifests = self._manifests[-MAX_MANIFESTS:]
        data = [m.to_dict() for m in self._manifests]
        _save_json(MANIFESTS_FILE, data)
        logger.debug("Saved %d manifests to disk", len(self._manifests))

    # -----------------------------------------------------------------------
    # Internal — file discovery helpers
    # -----------------------------------------------------------------------

    def _discover_dirs(self, directories: Optional[List[str]] = None) -> List[Path]:
        """Return list of data/ subdirectory Paths that exist.

        If *directories* is None, uses all BACKUP_TARGETS.
        Only returns directories that actually exist on disk.
        """
        targets = directories if directories else self.BACKUP_TARGETS
        result = []
        for name in targets:
            p = DATA_DIR / name
            if p.exists() and p.is_dir():
                result.append(p)
        return result

    def _collect_files(self, dirs: List[Path]) -> List[Path]:
        """Recursively collect all files inside the given directories."""
        files: List[Path] = []
        for d in dirs:
            if not d.exists():
                continue
            for root, _, filenames in os.walk(d):
                for fn in filenames:
                    fp = Path(root) / fn
                    if fp.is_file():
                        files.append(fp)
        return files

    def _total_size(self, files: List[Path]) -> int:
        """Compute total size in bytes of a list of files."""
        total = 0
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return total

    def _file_relative(self, file_path: Path) -> str:
        """Return the path of *file_path* relative to DATA_DIR."""
        try:
            return str(file_path.relative_to(DATA_DIR))
        except ValueError:
            return str(file_path)

    # -----------------------------------------------------------------------
    # Internal — archive creation
    # -----------------------------------------------------------------------

    def _create_zip(
        self, source_dirs: List[Path], output_path: Path
    ) -> Tuple[int, int]:
        """Create a ZIP archive of all files in *source_dirs*.

        Returns (file_count, compressed_size_bytes).
        """
        file_count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for d in source_dirs:
                if not d.exists():
                    continue
                for root, _, filenames in os.walk(d):
                    for fn in filenames:
                        fp = Path(root) / fn
                        if fp.is_file():
                            arcname = self._file_relative(fp)
                            try:
                                zf.write(fp, arcname)
                                file_count += 1
                            except (OSError, PermissionError) as exc:
                                logger.warning("Skipping %s: %s", fp, exc)
        compressed_size = output_path.stat().st_size if output_path.exists() else 0
        return file_count, compressed_size

    def _create_gzip_tar(
        self, source_dirs: List[Path], output_path: Path
    ) -> Tuple[int, int]:
        """Create a gzip-compressed tar archive of all files in *source_dirs*.

        Returns (file_count, compressed_size_bytes).
        """
        file_count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz", compresslevel=6) as tf:
            for d in source_dirs:
                if not d.exists():
                    continue
                for root, _, filenames in os.walk(d):
                    for fn in filenames:
                        fp = Path(root) / fn
                        if fp.is_file():
                            arcname = self._file_relative(fp)
                            try:
                                tf.add(str(fp), arcname=arcname)
                                file_count += 1
                            except (OSError, PermissionError) as exc:
                                logger.warning("Skipping %s: %s", fp, exc)
        compressed_size = output_path.stat().st_size if output_path.exists() else 0
        return file_count, compressed_size

    def _create_no_compression(
        self, source_dirs: List[Path], output_path: Path
    ) -> Tuple[int, int]:
        """Create a plain tar (no compression) of all files in *source_dirs*.

        Returns (file_count, archive_size_bytes).
        """
        file_count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w") as tf:
            for d in source_dirs:
                if not d.exists():
                    continue
                for root, _, filenames in os.walk(d):
                    for fn in filenames:
                        fp = Path(root) / fn
                        if fp.is_file():
                            arcname = self._file_relative(fp)
                            try:
                                tf.add(str(fp), arcname=arcname)
                                file_count += 1
                            except (OSError, PermissionError) as exc:
                                logger.warning("Skipping %s: %s", fp, exc)
        archive_size = output_path.stat().st_size if output_path.exists() else 0
        return file_count, archive_size

    # -----------------------------------------------------------------------
    # Internal — archive extraction
    # -----------------------------------------------------------------------

    def _extract_zip(self, archive_path: Path, target_dir: Path) -> int:
        """Extract a ZIP archive into *target_dir*.  Returns file count."""
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                zf.extract(info, target_dir)
                count += 1
        return count

    def _extract_tar(self, archive_path: Path, target_dir: Path) -> int:
        """Extract a tar (plain or gzip) into *target_dir*.  Returns file count."""
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        # Determine mode from suffix
        if str(archive_path).endswith(".tar.gz") or str(archive_path).endswith(".tgz"):
            mode = "r:gz"
        else:
            mode = "r:"
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    # Security: prevent path traversal
                    if member.name.startswith("/") or ".." in member.name:
                        logger.warning("Skipping suspicious path: %s", member.name)
                        continue
                    tf.extract(member, target_dir)
                    count += 1
        return count

    # -----------------------------------------------------------------------
    # Internal — checksum
    # -----------------------------------------------------------------------

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as fh:
            while True:
                chunk = fh.read(HASH_BUFFER_SIZE)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    # -----------------------------------------------------------------------
    # Internal — changed file detection (incremental)
    # -----------------------------------------------------------------------

    def _get_changed_files(self, since: str, directories: Optional[List[str]] = None) -> List[Path]:
        """Find all files modified after *since* (ISO-8601) in the target directories.

        Returns a list of absolute Paths that have been created or modified
        since the given timestamp.
        """
        since_dt = _parse_iso(since)
        if since_dt is None:
            # If we can't parse, return all files (treat as full backup)
            dirs = self._discover_dirs(directories)
            return self._collect_files(dirs)

        since_ts = since_dt.timestamp()
        dirs = self._discover_dirs(directories)
        changed: List[Path] = []
        for d in dirs:
            if not d.exists():
                continue
            for root, _, filenames in os.walk(d):
                for fn in filenames:
                    fp = Path(root) / fn
                    try:
                        if fp.is_file() and fp.stat().st_mtime > since_ts:
                            changed.append(fp)
                    except OSError:
                        pass
        return changed

    def _archive_extension(self, compression: CompressionType) -> str:
        """Return the appropriate file extension for the compression type."""
        if compression == CompressionType.ZIP:
            return ".zip"
        elif compression == CompressionType.GZIP:
            return ".tar.gz"
        else:
            return ".tar"

    def _create_archive(
        self,
        source_dirs: List[Path],
        output_path: Path,
        compression: CompressionType,
    ) -> Tuple[int, int]:
        """Dispatch to the correct archive creator.  Returns (file_count, archive_size)."""
        if compression == CompressionType.ZIP:
            return self._create_zip(source_dirs, output_path)
        elif compression == CompressionType.GZIP:
            return self._create_gzip_tar(source_dirs, output_path)
        else:
            return self._create_no_compression(source_dirs, output_path)

    def _extract_archive(
        self,
        archive_path: Path,
        target_dir: Path,
        compression: CompressionType,
    ) -> int:
        """Dispatch to the correct extractor.  Returns file count."""
        if compression == CompressionType.ZIP:
            return self._extract_zip(archive_path, target_dir)
        else:
            return self._extract_tar(archive_path, target_dir)

    # -----------------------------------------------------------------------
    # Create — Full Backup
    # -----------------------------------------------------------------------

    async def create_full_backup(
        self,
        description: str = "",
        compression: Optional[CompressionType] = None,
    ) -> BackupManifest:
        """Create a full backup of all data/ directories.

        Parameters:
            description: human-readable note for this backup
            compression: override default compression (ZIP, GZIP, NONE)

        Returns:
            BackupManifest describing the completed backup.
        """
        comp = compression if compression is not None else self._default_compression
        backup_id = str(uuid.uuid4())
        timestamp_slug = _now_utc().strftime("%Y%m%d-%H%M%S")
        archive_name = f"full-{timestamp_slug}-{_short_id(backup_id)}{self._archive_extension(comp)}"
        archive_path = self._archives_dir / archive_name

        # Create manifest (pending)
        dirs = self._discover_dirs()
        all_files = self._collect_files(dirs)
        total_size = self._total_size(all_files)
        dir_names = [d.name for d in dirs]

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            status=BackupStatus.IN_PROGRESS,
            created_at=_now_iso(),
            completed_at=None,
            source_dir=str(DATA_DIR),
            backup_path=str(archive_path),
            file_count=len(all_files),
            total_size_bytes=total_size,
            compressed_size_bytes=0,
            compression=comp,
            directories_included=dir_names,
            description=description,
            checksum=None,
            parent_backup_id=None,
            metadata={
                "hostname": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")),
                "backup_targets": self.BACKUP_TARGETS,
            },
        )
        self._manifests.append(manifest)
        self._save_manifests()

        logger.info(
            "Creating full backup %s — %d files, %s uncompressed, compression=%s",
            _short_id(backup_id), len(all_files), _human_size(total_size), comp.value,
        )

        try:
            file_count, compressed_size = self._create_archive(dirs, archive_path, comp)
            checksum = self._compute_checksum(archive_path)

            manifest.status = BackupStatus.COMPLETED
            manifest.completed_at = _now_iso()
            manifest.file_count = file_count
            manifest.compressed_size_bytes = compressed_size
            manifest.checksum = checksum

            logger.info(
                "Full backup %s completed — %d files, %s compressed, checksum=%s",
                _short_id(backup_id), file_count, _human_size(compressed_size), checksum[:16],
            )
        except Exception as exc:
            manifest.status = BackupStatus.FAILED
            manifest.completed_at = _now_iso()
            manifest.metadata["error"] = str(exc)
            logger.error("Full backup %s FAILED: %s", _short_id(backup_id), exc)
            self._save_manifests()
            raise

        self._save_manifests()
        self._auto_prune()
        return manifest

    def create_full_backup_sync(
        self,
        description: str = "",
        compression: Optional[CompressionType] = None,
    ) -> BackupManifest:
        """Synchronous wrapper for :meth:`create_full_backup`."""
        return _run_sync(self.create_full_backup(description=description, compression=compression))

    # -----------------------------------------------------------------------
    # Create — Incremental Backup
    # -----------------------------------------------------------------------

    async def create_incremental_backup(
        self,
        description: str = "",
        compression: Optional[CompressionType] = None,
    ) -> BackupManifest:
        """Create an incremental backup containing only files changed since the last backup.

        If no previous backup exists, falls back to a full backup.

        Parameters:
            description: human-readable note
            compression: override default compression

        Returns:
            BackupManifest describing the completed incremental backup.
        """
        # Find the most recent completed backup to use as baseline
        parent = self.get_latest()
        if parent is None:
            logger.info("No previous backup found — falling back to full backup")
            return await self.create_full_backup(description=description or "Auto-full (no parent)", compression=compression)

        comp = compression if compression is not None else self._default_compression
        backup_id = str(uuid.uuid4())
        timestamp_slug = _now_utc().strftime("%Y%m%d-%H%M%S")
        archive_name = f"incr-{timestamp_slug}-{_short_id(backup_id)}{self._archive_extension(comp)}"
        archive_path = self._archives_dir / archive_name

        # Find changed files since parent backup
        since = parent.completed_at or parent.created_at
        changed_files = self._get_changed_files(since)

        if not changed_files:
            logger.info("No files changed since last backup %s — creating empty incremental manifest", _short_id(parent.backup_id))
            manifest = BackupManifest(
                backup_id=backup_id,
                backup_type=BackupType.INCREMENTAL,
                status=BackupStatus.COMPLETED,
                created_at=_now_iso(),
                completed_at=_now_iso(),
                source_dir=str(DATA_DIR),
                backup_path="",
                file_count=0,
                total_size_bytes=0,
                compressed_size_bytes=0,
                compression=comp,
                directories_included=[],
                description=description or "No changes detected",
                checksum=None,
                parent_backup_id=parent.backup_id,
                metadata={"since": since, "parent_backup": _short_id(parent.backup_id)},
            )
            self._manifests.append(manifest)
            self._save_manifests()
            return manifest

        total_size = self._total_size(changed_files)
        # Determine which directories are touched
        dir_names_set: Set[str] = set()
        for f in changed_files:
            try:
                rel = f.relative_to(DATA_DIR)
                dir_names_set.add(rel.parts[0])
            except (ValueError, IndexError):
                pass

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            status=BackupStatus.IN_PROGRESS,
            created_at=_now_iso(),
            completed_at=None,
            source_dir=str(DATA_DIR),
            backup_path=str(archive_path),
            file_count=len(changed_files),
            total_size_bytes=total_size,
            compressed_size_bytes=0,
            compression=comp,
            directories_included=sorted(dir_names_set),
            description=description,
            checksum=None,
            parent_backup_id=parent.backup_id,
            metadata={
                "since": since,
                "parent_backup": _short_id(parent.backup_id),
                "changed_file_count": len(changed_files),
            },
        )
        self._manifests.append(manifest)
        self._save_manifests()

        logger.info(
            "Creating incremental backup %s — %d changed files since %s, %s",
            _short_id(backup_id), len(changed_files), since, _human_size(total_size),
        )

        try:
            # Write changed files into archive directly (not via dirs)
            file_count, compressed_size = self._create_incremental_archive(
                changed_files, archive_path, comp
            )
            checksum = self._compute_checksum(archive_path)

            manifest.status = BackupStatus.COMPLETED
            manifest.completed_at = _now_iso()
            manifest.file_count = file_count
            manifest.compressed_size_bytes = compressed_size
            manifest.checksum = checksum

            logger.info(
                "Incremental backup %s completed — %d files, %s",
                _short_id(backup_id), file_count, _human_size(compressed_size),
            )
        except Exception as exc:
            manifest.status = BackupStatus.FAILED
            manifest.completed_at = _now_iso()
            manifest.metadata["error"] = str(exc)
            logger.error("Incremental backup %s FAILED: %s", _short_id(backup_id), exc)
            self._save_manifests()
            raise

        self._save_manifests()
        self._auto_prune()
        return manifest

    def _create_incremental_archive(
        self,
        files: List[Path],
        output_path: Path,
        compression: CompressionType,
    ) -> Tuple[int, int]:
        """Create an archive containing only specific files.  Returns (count, size)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_count = 0

        if compression == CompressionType.ZIP:
            with zipfile.ZipFile(
                output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as zf:
                for fp in files:
                    if fp.is_file():
                        arcname = self._file_relative(fp)
                        try:
                            zf.write(fp, arcname)
                            file_count += 1
                        except (OSError, PermissionError) as exc:
                            logger.warning("Skipping %s: %s", fp, exc)
        elif compression == CompressionType.GZIP:
            with tarfile.open(output_path, "w:gz", compresslevel=6) as tf:
                for fp in files:
                    if fp.is_file():
                        arcname = self._file_relative(fp)
                        try:
                            tf.add(str(fp), arcname=arcname)
                            file_count += 1
                        except (OSError, PermissionError) as exc:
                            logger.warning("Skipping %s: %s", fp, exc)
        else:
            with tarfile.open(output_path, "w") as tf:
                for fp in files:
                    if fp.is_file():
                        arcname = self._file_relative(fp)
                        try:
                            tf.add(str(fp), arcname=arcname)
                            file_count += 1
                        except (OSError, PermissionError) as exc:
                            logger.warning("Skipping %s: %s", fp, exc)

        compressed_size = output_path.stat().st_size if output_path.exists() else 0
        return file_count, compressed_size

    def create_incremental_backup_sync(
        self,
        description: str = "",
        compression: Optional[CompressionType] = None,
    ) -> BackupManifest:
        """Synchronous wrapper for :meth:`create_incremental_backup`."""
        return _run_sync(self.create_incremental_backup(description=description, compression=compression))

    # -----------------------------------------------------------------------
    # Create — Selective Backup
    # -----------------------------------------------------------------------

    async def create_selective_backup(
        self,
        directories: List[str],
        description: str = "",
        compression: Optional[CompressionType] = None,
    ) -> BackupManifest:
        """Back up only the specified data/ subdirectories.

        Parameters:
            directories: list of directory names (e.g. ["accounts", "auth"])
            description: human-readable note
            compression: override default compression

        Returns:
            BackupManifest describing the completed selective backup.

        Raises:
            ValueError: if no valid directories are specified.
        """
        # Validate directories
        valid_dirs = [d for d in directories if d in self.BACKUP_TARGETS]
        if not valid_dirs:
            raise ValueError(
                f"No valid backup targets in {directories}. "
                f"Valid targets: {self.BACKUP_TARGETS}"
            )

        invalid = set(directories) - set(valid_dirs)
        if invalid:
            logger.warning("Ignoring unknown directories: %s", invalid)

        comp = compression if compression is not None else self._default_compression
        backup_id = str(uuid.uuid4())
        timestamp_slug = _now_utc().strftime("%Y%m%d-%H%M%S")
        dir_slug = "-".join(valid_dirs[:3])
        if len(valid_dirs) > 3:
            dir_slug += f"-plus{len(valid_dirs) - 3}"
        archive_name = f"sel-{dir_slug}-{timestamp_slug}-{_short_id(backup_id)}{self._archive_extension(comp)}"
        archive_path = self._archives_dir / archive_name

        dirs = self._discover_dirs(valid_dirs)
        all_files = self._collect_files(dirs)
        total_size = self._total_size(all_files)

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.SELECTIVE,
            status=BackupStatus.IN_PROGRESS,
            created_at=_now_iso(),
            completed_at=None,
            source_dir=str(DATA_DIR),
            backup_path=str(archive_path),
            file_count=len(all_files),
            total_size_bytes=total_size,
            compressed_size_bytes=0,
            compression=comp,
            directories_included=valid_dirs,
            description=description,
            checksum=None,
            parent_backup_id=None,
            metadata={"requested_dirs": directories, "valid_dirs": valid_dirs},
        )
        self._manifests.append(manifest)
        self._save_manifests()

        logger.info(
            "Creating selective backup %s — dirs=%s, %d files, %s",
            _short_id(backup_id), valid_dirs, len(all_files), _human_size(total_size),
        )

        try:
            file_count, compressed_size = self._create_archive(dirs, archive_path, comp)
            checksum = self._compute_checksum(archive_path)

            manifest.status = BackupStatus.COMPLETED
            manifest.completed_at = _now_iso()
            manifest.file_count = file_count
            manifest.compressed_size_bytes = compressed_size
            manifest.checksum = checksum

            logger.info(
                "Selective backup %s completed — %d files, %s",
                _short_id(backup_id), file_count, _human_size(compressed_size),
            )
        except Exception as exc:
            manifest.status = BackupStatus.FAILED
            manifest.completed_at = _now_iso()
            manifest.metadata["error"] = str(exc)
            logger.error("Selective backup %s FAILED: %s", _short_id(backup_id), exc)
            self._save_manifests()
            raise

        self._save_manifests()
        self._auto_prune()
        return manifest

    def create_selective_backup_sync(
        self,
        directories: List[str],
        description: str = "",
        compression: Optional[CompressionType] = None,
    ) -> BackupManifest:
        """Synchronous wrapper for :meth:`create_selective_backup`."""
        return _run_sync(
            self.create_selective_backup(
                directories=directories, description=description, compression=compression
            )
        )

    # -----------------------------------------------------------------------
    # Create — Snapshot (metadata only)
    # -----------------------------------------------------------------------

    async def create_snapshot(
        self,
        description: str = "",
    ) -> BackupManifest:
        """Create a snapshot: metadata and file listing only — no data copied.

        Snapshots are lightweight checkpoints that record the current state
        (directory structure, file names, sizes, timestamps) without copying
        any actual data.  Useful for tracking state between real backups.

        Returns:
            BackupManifest describing the snapshot.
        """
        backup_id = str(uuid.uuid4())
        timestamp_slug = _now_utc().strftime("%Y%m%d-%H%M%S")
        snapshot_name = f"snap-{timestamp_slug}-{_short_id(backup_id)}.json"
        snapshot_path = self._snapshots_dir / snapshot_name

        dirs = self._discover_dirs()
        all_files = self._collect_files(dirs)
        total_size = self._total_size(all_files)
        dir_names = [d.name for d in dirs]

        # Build file listing
        file_listing: List[Dict[str, Any]] = []
        for fp in all_files:
            try:
                st = fp.stat()
                file_listing.append({
                    "path": self._file_relative(fp),
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                })
            except OSError:
                file_listing.append({
                    "path": self._file_relative(fp),
                    "size": 0,
                    "mtime": None,
                })

        snapshot_data = {
            "backup_id": backup_id,
            "created_at": _now_iso(),
            "directories": dir_names,
            "file_count": len(file_listing),
            "total_size_bytes": total_size,
            "files": file_listing,
            "description": description,
        }
        _save_json(snapshot_path, snapshot_data)

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=BackupType.SNAPSHOT,
            status=BackupStatus.COMPLETED,
            created_at=_now_iso(),
            completed_at=_now_iso(),
            source_dir=str(DATA_DIR),
            backup_path=str(snapshot_path),
            file_count=len(file_listing),
            total_size_bytes=total_size,
            compressed_size_bytes=snapshot_path.stat().st_size if snapshot_path.exists() else 0,
            compression=CompressionType.NONE,
            directories_included=dir_names,
            description=description,
            checksum=None,
            parent_backup_id=None,
            metadata={
                "snapshot_type": "metadata_only",
                "dir_count": len(dir_names),
            },
        )
        self._manifests.append(manifest)
        self._save_manifests()

        logger.info(
            "Snapshot %s created — %d files in %d dirs, %s total",
            _short_id(backup_id), len(file_listing), len(dir_names), _human_size(total_size),
        )
        return manifest

    def create_snapshot_sync(self, description: str = "") -> BackupManifest:
        """Synchronous wrapper for :meth:`create_snapshot`."""
        return _run_sync(self.create_snapshot(description=description))

    # -----------------------------------------------------------------------
    # Restore
    # -----------------------------------------------------------------------

    async def restore(
        self,
        backup_id: str,
        target_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Restore a backup.

        Parameters:
            backup_id: UUID of the backup to restore
            target_dir: where to extract (defaults to DATA_DIR)
            dry_run: if True, report what would be restored without writing

        Returns:
            Dict with keys: backup_id, target_dir, file_count, status, files (if dry_run)
        """
        manifest = self.get_backup(backup_id)
        if manifest.backup_type == BackupType.SNAPSHOT:
            return {
                "backup_id": backup_id,
                "status": "skipped",
                "reason": "Snapshots contain metadata only — nothing to restore",
            }

        archive_path = Path(manifest.backup_path)
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Backup archive not found: {archive_path}"
            )

        target = target_dir if target_dir is not None else DATA_DIR

        if dry_run:
            # List contents without extracting
            file_list = self._list_archive_contents(archive_path, manifest.compression)
            return {
                "backup_id": backup_id,
                "target_dir": str(target),
                "file_count": len(file_list),
                "status": "dry_run",
                "files": file_list,
                "total_size_bytes": manifest.total_size_bytes,
                "compressed_size_bytes": manifest.compressed_size_bytes,
            }

        # Mark as restoring
        manifest.status = BackupStatus.RESTORING
        self._save_manifests()

        logger.info(
            "Restoring backup %s (%s) to %s — %d files",
            _short_id(backup_id), manifest.backup_type.value, target, manifest.file_count,
        )

        try:
            restored_count = self._extract_archive(archive_path, target, manifest.compression)

            logger.info(
                "Restore of %s completed — %d files extracted to %s",
                _short_id(backup_id), restored_count, target,
            )
            # Revert status to completed (do not keep RESTORING)
            manifest.status = BackupStatus.COMPLETED
            self._save_manifests()

            return {
                "backup_id": backup_id,
                "target_dir": str(target),
                "file_count": restored_count,
                "status": "restored",
            }
        except Exception as exc:
            manifest.status = BackupStatus.FAILED
            manifest.metadata["restore_error"] = str(exc)
            self._save_manifests()
            logger.error("Restore of %s FAILED: %s", _short_id(backup_id), exc)
            raise

    def restore_sync(
        self,
        backup_id: str,
        target_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`restore`."""
        return _run_sync(self.restore(backup_id=backup_id, target_dir=target_dir, dry_run=dry_run))

    # -----------------------------------------------------------------------
    # Restore — Selective
    # -----------------------------------------------------------------------

    async def restore_selective(
        self,
        backup_id: str,
        directories: List[str],
        dry_run: bool = False,
        target_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Restore only specific directories from a backup.

        Parameters:
            backup_id: UUID of the backup
            directories: directory names to restore (e.g. ["accounts", "auth"])
            dry_run: if True, report without extracting
            target_dir: where to extract (defaults to DATA_DIR)

        Returns:
            Dict with restoration details.
        """
        manifest = self.get_backup(backup_id)
        if manifest.backup_type == BackupType.SNAPSHOT:
            return {
                "backup_id": backup_id,
                "status": "skipped",
                "reason": "Snapshots contain metadata only — nothing to restore",
            }

        archive_path = Path(manifest.backup_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")

        target = target_dir if target_dir is not None else DATA_DIR

        # Filter archive contents to only the requested directories
        all_files = self._list_archive_contents(archive_path, manifest.compression)
        matching: List[str] = []
        dir_prefixes = [d + "/" for d in directories] + [d + "\\" for d in directories]
        for f in all_files:
            for prefix in dir_prefixes:
                if f.startswith(prefix):
                    matching.append(f)
                    break

        if dry_run:
            return {
                "backup_id": backup_id,
                "target_dir": str(target),
                "directories_requested": directories,
                "file_count": len(matching),
                "status": "dry_run",
                "files": matching,
            }

        logger.info(
            "Selective restore of %s — dirs=%s, %d matching files",
            _short_id(backup_id), directories, len(matching),
        )

        try:
            restored_count = self._extract_selective(
                archive_path, target, manifest.compression, matching
            )
            logger.info(
                "Selective restore of %s completed — %d files",
                _short_id(backup_id), restored_count,
            )
            return {
                "backup_id": backup_id,
                "target_dir": str(target),
                "directories_restored": directories,
                "file_count": restored_count,
                "status": "restored",
            }
        except Exception as exc:
            logger.error("Selective restore of %s FAILED: %s", _short_id(backup_id), exc)
            raise

    def restore_selective_sync(
        self,
        backup_id: str,
        directories: List[str],
        dry_run: bool = False,
        target_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`restore_selective`."""
        return _run_sync(
            self.restore_selective(
                backup_id=backup_id, directories=directories,
                dry_run=dry_run, target_dir=target_dir,
            )
        )

    # -----------------------------------------------------------------------
    # Internal — archive listing and selective extraction
    # -----------------------------------------------------------------------

    def _list_archive_contents(
        self, archive_path: Path, compression: CompressionType
    ) -> List[str]:
        """List all file paths inside an archive."""
        entries: List[str] = []
        if compression == CompressionType.ZIP:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        entries.append(info.filename)
        else:
            mode = "r:gz" if compression == CompressionType.GZIP else "r:"
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        entries.append(member.name)
        return entries

    def _extract_selective(
        self,
        archive_path: Path,
        target_dir: Path,
        compression: CompressionType,
        matching_paths: List[str],
    ) -> int:
        """Extract only the files whose archive paths are in *matching_paths*."""
        target_dir.mkdir(parents=True, exist_ok=True)
        matching_set = set(matching_paths)
        count = 0

        if compression == CompressionType.ZIP:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir() and info.filename in matching_set:
                        zf.extract(info, target_dir)
                        count += 1
        else:
            mode = "r:gz" if compression == CompressionType.GZIP else "r:"
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if member.isfile() and member.name in matching_set:
                        # Security check
                        if member.name.startswith("/") or ".." in member.name:
                            continue
                        tf.extract(member, target_dir)
                        count += 1

        return count

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    async def verify(self, backup_id: str) -> Dict[str, Any]:
        """Verify the integrity of a backup.

        Checks:
            - Archive file exists
            - SHA-256 checksum matches manifest
            - Archive is readable and not corrupt
            - File count matches manifest

        Returns:
            Dict with verification results.
        """
        manifest = self.get_backup(backup_id)
        result: Dict[str, Any] = {
            "backup_id": backup_id,
            "backup_type": manifest.backup_type.value,
            "status": "unknown",
            "checks": {},
        }

        # Snapshots: just verify the JSON file exists and is readable
        if manifest.backup_type == BackupType.SNAPSHOT:
            snapshot_path = Path(manifest.backup_path)
            exists = snapshot_path.exists()
            result["checks"]["file_exists"] = exists
            if exists:
                try:
                    data = _load_json(snapshot_path, default=None)
                    readable = data is not None
                except Exception:
                    readable = False
                result["checks"]["readable"] = readable
                result["status"] = "verified" if (exists and readable) else "failed"
            else:
                result["checks"]["readable"] = False
                result["status"] = "failed"
            if result["status"] == "verified":
                manifest.status = BackupStatus.VERIFIED
                self._save_manifests()
            return result

        # Archive-based backups
        archive_path = Path(manifest.backup_path)

        # Check 1: file exists
        exists = archive_path.exists()
        result["checks"]["file_exists"] = exists
        if not exists:
            result["status"] = "failed"
            result["checks"]["reason"] = f"Archive not found: {archive_path}"
            return result

        # Check 2: file size > 0
        size = archive_path.stat().st_size
        result["checks"]["file_size"] = size
        result["checks"]["size_nonzero"] = size > 0

        # Check 3: checksum
        if manifest.checksum:
            current_checksum = self._compute_checksum(archive_path)
            checksum_match = current_checksum == manifest.checksum
            result["checks"]["checksum_match"] = checksum_match
            result["checks"]["expected_checksum"] = manifest.checksum[:16] + "..."
            result["checks"]["actual_checksum"] = current_checksum[:16] + "..."
        else:
            checksum_match = True  # no checksum to verify
            result["checks"]["checksum_match"] = "no_checksum_stored"

        # Check 4: archive is readable and file count
        try:
            file_list = self._list_archive_contents(archive_path, manifest.compression)
            archive_readable = True
            actual_file_count = len(file_list)
        except Exception as exc:
            archive_readable = False
            actual_file_count = 0
            result["checks"]["read_error"] = str(exc)

        result["checks"]["archive_readable"] = archive_readable
        result["checks"]["manifest_file_count"] = manifest.file_count
        result["checks"]["actual_file_count"] = actual_file_count

        # For incremental with 0 files, it is OK if manifest says 0
        if manifest.file_count == 0 and manifest.backup_path == "":
            file_count_ok = True
        else:
            file_count_ok = actual_file_count == manifest.file_count
        result["checks"]["file_count_match"] = file_count_ok

        # Overall verdict
        all_ok = (
            exists
            and size > 0
            and (checksum_match if isinstance(checksum_match, bool) else True)
            and archive_readable
            and file_count_ok
        )
        result["status"] = "verified" if all_ok else "failed"

        if all_ok:
            manifest.status = BackupStatus.VERIFIED
            self._save_manifests()
            logger.info("Backup %s verified successfully", _short_id(backup_id))
        else:
            logger.warning("Backup %s verification FAILED: %s", _short_id(backup_id), result["checks"])

        return result

    def verify_sync(self, backup_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`verify`."""
        return _run_sync(self.verify(backup_id=backup_id))

    # -----------------------------------------------------------------------
    # Management — list, get, delete, prune
    # -----------------------------------------------------------------------

    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        limit: int = 0,
    ) -> List[BackupManifest]:
        """Return manifests, optionally filtered by type.

        Parameters:
            backup_type: filter to this type, or None for all
            limit: max results (0 = unlimited), newest first

        Returns:
            List of BackupManifest, sorted newest-first.
        """
        results = self._manifests[:]
        if backup_type is not None:
            results = [m for m in results if m.backup_type == backup_type]
        # Newest first
        results.sort(key=lambda m: m.created_at, reverse=True)
        if limit > 0:
            results = results[:limit]
        return results

    def get_backup(self, backup_id: str) -> BackupManifest:
        """Retrieve a specific backup manifest by ID.

        Raises:
            KeyError: if no backup with that ID exists.
        """
        for m in self._manifests:
            if m.backup_id == backup_id:
                return m
        # Also try short-id matching
        for m in self._manifests:
            if m.backup_id.startswith(backup_id):
                return m
        raise KeyError(f"No backup found with ID '{backup_id}'")

    def delete_backup(self, backup_id: str) -> None:
        """Delete a backup: remove the archive file and its manifest entry.

        Raises:
            KeyError: if no backup with that ID exists.
        """
        manifest = self.get_backup(backup_id)
        # Remove the archive/snapshot file
        backup_path = Path(manifest.backup_path) if manifest.backup_path else None
        if backup_path and backup_path.exists():
            try:
                backup_path.unlink()
                logger.info("Deleted archive: %s", backup_path)
            except OSError as exc:
                logger.warning("Could not delete archive %s: %s", backup_path, exc)

        # Remove from manifest list
        self._manifests = [m for m in self._manifests if m.backup_id != manifest.backup_id]
        self._save_manifests()
        logger.info("Deleted backup %s (%s)", _short_id(backup_id), manifest.backup_type.value)

    def prune_old(self, keep: int = DEFAULT_KEEP_BACKUPS) -> int:
        """Delete the oldest backups, keeping at most *keep* completed backups.

        Parameters:
            keep: number of completed backups to retain

        Returns:
            Number of backups deleted.
        """
        # Sort oldest-first by created_at
        completed = [m for m in self._manifests if m.status in (
            BackupStatus.COMPLETED, BackupStatus.VERIFIED
        )]
        completed.sort(key=lambda m: m.created_at)

        to_delete = completed[:-keep] if len(completed) > keep else []
        deleted_count = 0
        for m in to_delete:
            try:
                self.delete_backup(m.backup_id)
                deleted_count += 1
            except Exception as exc:
                logger.warning("Could not prune backup %s: %s", _short_id(m.backup_id), exc)

        if deleted_count > 0:
            logger.info("Pruned %d old backups, keeping %d", deleted_count, keep)
        return deleted_count

    def _auto_prune(self) -> None:
        """Automatically prune if we exceed _max_backups."""
        completed = [m for m in self._manifests if m.status in (
            BackupStatus.COMPLETED, BackupStatus.VERIFIED
        )]
        if len(completed) > self._max_backups:
            self.prune_old(keep=self._max_backups)

    def get_latest(
        self,
        backup_type: Optional[BackupType] = None,
    ) -> Optional[BackupManifest]:
        """Return the most recent completed backup, optionally filtered by type.

        Returns None if no completed backups exist.
        """
        candidates = [
            m for m in self._manifests
            if m.status in (BackupStatus.COMPLETED, BackupStatus.VERIFIED)
        ]
        if backup_type is not None:
            candidates = [m for m in candidates if m.backup_type == backup_type]
        if not candidates:
            return None
        candidates.sort(key=lambda m: m.created_at, reverse=True)
        return candidates[0]

    # -----------------------------------------------------------------------
    # Analytics — stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics about all backups.

        Returns dict with:
            total_backups, completed, failed, by_type counts,
            total_compressed_size, avg_compressed_size,
            oldest_backup, newest_backup, etc.
        """
        total = len(self._manifests)
        completed = [m for m in self._manifests if m.status in (
            BackupStatus.COMPLETED, BackupStatus.VERIFIED
        )]
        failed = [m for m in self._manifests if m.status == BackupStatus.FAILED]

        by_type: Dict[str, int] = {}
        for bt in BackupType:
            by_type[bt.value] = len([m for m in self._manifests if m.backup_type == bt])

        by_status: Dict[str, int] = {}
        for bs in BackupStatus:
            by_status[bs.value] = len([m for m in self._manifests if m.status == bs])

        total_compressed = sum(m.compressed_size_bytes for m in completed)
        total_uncompressed = sum(m.total_size_bytes for m in completed)
        avg_compressed = total_compressed // len(completed) if completed else 0
        total_file_count = sum(m.file_count for m in completed)

        # Compression ratio
        if total_uncompressed > 0:
            compression_ratio = 1.0 - (total_compressed / total_uncompressed)
        else:
            compression_ratio = 0.0

        oldest = min((m.created_at for m in self._manifests), default=None)
        newest = max((m.created_at for m in self._manifests), default=None)

        return {
            "total_backups": total,
            "completed": len(completed),
            "failed": len(failed),
            "by_type": by_type,
            "by_status": by_status,
            "total_compressed_size_bytes": total_compressed,
            "total_compressed_size": _human_size(total_compressed),
            "total_uncompressed_size_bytes": total_uncompressed,
            "total_uncompressed_size": _human_size(total_uncompressed),
            "avg_compressed_size_bytes": avg_compressed,
            "avg_compressed_size": _human_size(avg_compressed),
            "compression_ratio": f"{compression_ratio:.1%}",
            "total_file_count": total_file_count,
            "oldest_backup": oldest,
            "newest_backup": newest,
            "max_backups_setting": self._max_backups,
            "default_compression": self._default_compression.value,
        }

    # -----------------------------------------------------------------------
    # Analytics — disk usage
    # -----------------------------------------------------------------------

    def get_disk_usage(self) -> Dict[str, Any]:
        """Return disk usage breakdown for the backup directory.

        Returns dict with total backup dir size, per-backup sizes, and
        archive vs snapshot breakdown.
        """
        # Total backup directory size
        total_dir_size = 0
        for root, _, filenames in os.walk(BACKUP_DATA_DIR):
            for fn in filenames:
                fp = Path(root) / fn
                try:
                    total_dir_size += fp.stat().st_size
                except OSError:
                    pass

        # Per-backup sizes
        per_backup: List[Dict[str, Any]] = []
        for m in self._manifests:
            bp = Path(m.backup_path) if m.backup_path else None
            actual_size = 0
            if bp and bp.exists():
                try:
                    actual_size = bp.stat().st_size
                except OSError:
                    pass
            per_backup.append({
                "backup_id": _short_id(m.backup_id),
                "type": m.backup_type.value,
                "status": m.status.value,
                "compressed_size": _human_size(m.compressed_size_bytes),
                "actual_disk_size": _human_size(actual_size),
                "actual_disk_size_bytes": actual_size,
                "created_at": m.created_at,
            })

        # Archive vs snapshot subtotals
        archives_size = 0
        snapshots_size = 0
        for root, _, filenames in os.walk(self._archives_dir):
            for fn in filenames:
                try:
                    archives_size += (Path(root) / fn).stat().st_size
                except OSError:
                    pass
        for root, _, filenames in os.walk(self._snapshots_dir):
            for fn in filenames:
                try:
                    snapshots_size += (Path(root) / fn).stat().st_size
                except OSError:
                    pass

        return {
            "backup_dir": str(BACKUP_DATA_DIR),
            "total_dir_size_bytes": total_dir_size,
            "total_dir_size": _human_size(total_dir_size),
            "archives_size_bytes": archives_size,
            "archives_size": _human_size(archives_size),
            "snapshots_size_bytes": snapshots_size,
            "snapshots_size": _human_size(snapshots_size),
            "manifests_size_bytes": MANIFESTS_FILE.stat().st_size if MANIFESTS_FILE.exists() else 0,
            "backup_count": len(self._manifests),
            "per_backup": per_backup,
        }

    # -----------------------------------------------------------------------
    # Analytics — diff with current
    # -----------------------------------------------------------------------

    def diff_with_current(self, backup_id: str) -> Dict[str, Any]:
        """Compare a backup's state with the current data/ directory.

        For archive-based backups, lists files in the archive and compares
        with current files on disk.  For snapshots, uses the stored file
        listing.

        Returns dict with: added, removed, modified file lists.
        """
        manifest = self.get_backup(backup_id)

        # Get backup file listing
        if manifest.backup_type == BackupType.SNAPSHOT:
            backup_files = self._get_snapshot_file_map(manifest)
        else:
            backup_files = self._get_archive_file_map(manifest)

        # Get current file listing
        current_files: Dict[str, Dict[str, Any]] = {}
        dirs = self._discover_dirs()
        for fp in self._collect_files(dirs):
            rel = self._file_relative(fp)
            try:
                st = fp.stat()
                current_files[rel] = {
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
            except OSError:
                current_files[rel] = {"size": 0, "mtime": None}

        backup_paths = set(backup_files.keys())
        current_paths = set(current_files.keys())

        added = sorted(current_paths - backup_paths)
        removed = sorted(backup_paths - current_paths)
        common = backup_paths & current_paths

        modified: List[str] = []
        for path in sorted(common):
            b = backup_files[path]
            c = current_files[path]
            # Compare by size (mtime comparison is unreliable across restores)
            if b.get("size") != c.get("size"):
                modified.append(path)

        return {
            "backup_id": _short_id(backup_id),
            "backup_type": manifest.backup_type.value,
            "backup_created_at": manifest.created_at,
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "unchanged_count": len(common) - len(modified),
            "added": added[:100],  # cap for display
            "removed": removed[:100],
            "modified": modified[:100],
            "total_current_files": len(current_files),
            "total_backup_files": len(backup_files),
        }

    def _get_snapshot_file_map(self, manifest: BackupManifest) -> Dict[str, Dict[str, Any]]:
        """Load file map from a snapshot JSON."""
        snapshot_path = Path(manifest.backup_path)
        if not snapshot_path.exists():
            return {}
        data = _load_json(snapshot_path, default={})
        files = data.get("files", [])
        result: Dict[str, Dict[str, Any]] = {}
        for entry in files:
            result[entry["path"]] = {
                "size": entry.get("size", 0),
                "mtime": entry.get("mtime"),
            }
        return result

    def _get_archive_file_map(self, manifest: BackupManifest) -> Dict[str, Dict[str, Any]]:
        """Build file map from an archive (ZIP or tar)."""
        archive_path = Path(manifest.backup_path)
        if not archive_path.exists():
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        if manifest.compression == CompressionType.ZIP:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        result[info.filename] = {
                            "size": info.file_size,
                            "mtime": None,
                        }
        else:
            mode = "r:gz" if manifest.compression == CompressionType.GZIP else "r:"
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        result[member.name] = {
                            "size": member.size,
                            "mtime": None,
                        }
        return result

    # -----------------------------------------------------------------------
    # Summary / repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BackupManager(backups={len(self._manifests)}, "
            f"max={self._max_backups}, compression={self._default_compression.value})"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Return the module-level BackupManager singleton, creating it on first call."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


# ---------------------------------------------------------------------------
# CLI — main()
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI use."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def _print_manifest(m: BackupManifest, detailed: bool = False) -> None:
    """Print a single manifest to stdout."""
    status_icon = {
        BackupStatus.COMPLETED: "[OK]",
        BackupStatus.VERIFIED: "[OK+]",
        BackupStatus.FAILED: "[FAIL]",
        BackupStatus.IN_PROGRESS: "[...]",
        BackupStatus.PENDING: "[...]",
        BackupStatus.RESTORING: "[RST]",
    }.get(m.status, "[?]")

    print(
        f"  {status_icon} {_short_id(m.backup_id)}  "
        f"{m.backup_type.value:<12}  "
        f"{m.file_count:>6} files  "
        f"{_human_size(m.compressed_size_bytes):>10}  "
        f"{m.created_at[:19]}  "
        f"{m.description[:40]}"
    )
    if detailed:
        print(f"         ID: {m.backup_id}")
        print(f"         Status: {m.status.value}")
        print(f"         Source: {m.source_dir}")
        print(f"         Archive: {m.backup_path}")
        print(f"         Uncompressed: {_human_size(m.total_size_bytes)}")
        print(f"         Compression: {m.compression.value}")
        print(f"         Dirs: {', '.join(m.directories_included)}")
        if m.checksum:
            print(f"         Checksum: {m.checksum[:32]}...")
        if m.parent_backup_id:
            print(f"         Parent: {_short_id(m.parent_backup_id)}")
        if m.completed_at:
            print(f"         Completed: {m.completed_at[:19]}")
        print()


def main() -> None:
    """CLI entry point with subcommands for backup management."""
    parser = argparse.ArgumentParser(
        prog="backup_manager",
        description="OpenClaw Empire — Backup Manager for all data/ directories",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # -- create --
    p_create = sub.add_parser("create", help="Create a new backup")
    p_create.add_argument(
        "--type", "-t",
        choices=["full", "incremental", "selective"],
        default="full",
        help="Backup type (default: full)",
    )
    p_create.add_argument(
        "--dirs", "-d",
        nargs="+",
        help="Directories to back up (for selective type)",
    )
    p_create.add_argument(
        "--description", "--desc",
        default="",
        help="Description for this backup",
    )
    p_create.add_argument(
        "--compression", "-c",
        choices=["zip", "gzip", "none"],
        default=None,
        help="Compression type (default: zip)",
    )

    # -- snapshot --
    p_snap = sub.add_parser("snapshot", help="Create a metadata-only snapshot")
    p_snap.add_argument("--description", "--desc", default="", help="Description")

    # -- restore --
    p_restore = sub.add_parser("restore", help="Restore a backup")
    p_restore.add_argument("--id", required=True, help="Backup ID (full or prefix)")
    p_restore.add_argument(
        "--target-dir", default=None, help="Target directory (default: data/)"
    )
    p_restore.add_argument(
        "--dry-run", action="store_true", help="Preview without extracting"
    )
    p_restore.add_argument(
        "--dirs", "-d", nargs="+",
        help="Restore only specific directories (selective restore)",
    )

    # -- verify --
    p_verify = sub.add_parser("verify", help="Verify backup integrity")
    p_verify.add_argument("--id", required=True, help="Backup ID")

    # -- list --
    p_list = sub.add_parser("list", help="List backups")
    p_list.add_argument(
        "--type", "-t",
        choices=["full", "incremental", "selective", "snapshot"],
        default=None,
        help="Filter by backup type",
    )
    p_list.add_argument(
        "--limit", "-n", type=int, default=0, help="Max results"
    )
    p_list.add_argument(
        "--detailed", action="store_true", help="Show full details"
    )

    # -- delete --
    p_delete = sub.add_parser("delete", help="Delete a backup")
    p_delete.add_argument("--id", required=True, help="Backup ID")

    # -- prune --
    p_prune = sub.add_parser("prune", help="Prune old backups")
    p_prune.add_argument(
        "--keep", "-k", type=int, default=DEFAULT_KEEP_BACKUPS,
        help=f"Number of backups to keep (default: {DEFAULT_KEEP_BACKUPS})",
    )

    # -- stats --
    sub.add_parser("stats", help="Show backup statistics")

    # -- diff --
    p_diff = sub.add_parser("diff", help="Diff a backup with current state")
    p_diff.add_argument("--id", required=True, help="Backup ID")

    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    bm = get_backup_manager()

    # -- Dispatch --

    if args.command == "create":
        comp = CompressionType(args.compression) if args.compression else None

        if args.type == "full":
            manifest = bm.create_full_backup_sync(
                description=args.description, compression=comp,
            )
        elif args.type == "incremental":
            manifest = bm.create_incremental_backup_sync(
                description=args.description, compression=comp,
            )
        elif args.type == "selective":
            if not args.dirs:
                print("ERROR: --dirs is required for selective backup")
                print(f"  Valid targets: {', '.join(BACKUP_TARGETS)}")
                sys.exit(1)
            manifest = bm.create_selective_backup_sync(
                directories=args.dirs, description=args.description, compression=comp,
            )
        else:
            print(f"ERROR: Unknown backup type '{args.type}'")
            sys.exit(1)

        print(f"\nBackup created successfully:")
        _print_manifest(manifest, detailed=True)

    elif args.command == "snapshot":
        manifest = bm.create_snapshot_sync(description=args.description)
        print(f"\nSnapshot created:")
        _print_manifest(manifest, detailed=True)

    elif args.command == "restore":
        target_dir = Path(args.target_dir) if args.target_dir else None

        if args.dirs:
            result = bm.restore_selective_sync(
                backup_id=args.id,
                directories=args.dirs,
                dry_run=args.dry_run,
                target_dir=target_dir,
            )
        else:
            result = bm.restore_sync(
                backup_id=args.id,
                target_dir=target_dir,
                dry_run=args.dry_run,
            )

        print(f"\nRestore {'(DRY RUN) ' if args.dry_run else ''}result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Backup ID: {result.get('backup_id')}")
        print(f"  Target: {result.get('target_dir', 'N/A')}")
        print(f"  Files: {result.get('file_count', 0)}")

        if args.dry_run and "files" in result:
            print(f"\n  Files that would be restored ({len(result['files'])}):")
            for f in result["files"][:50]:
                print(f"    {f}")
            if len(result["files"]) > 50:
                print(f"    ... and {len(result['files']) - 50} more")

    elif args.command == "verify":
        result = bm.verify_sync(backup_id=args.id)
        print(f"\nVerification result for {_short_id(args.id)}:")
        print(f"  Status: {result['status'].upper()}")
        for check, value in result.get("checks", {}).items():
            icon = "[PASS]" if value is True else ("[FAIL]" if value is False else "[INFO]")
            print(f"  {icon} {check}: {value}")

    elif args.command == "list":
        bt = BackupType(args.type) if args.type else None
        backups = bm.list_backups(backup_type=bt, limit=args.limit)

        if not backups:
            print("\nNo backups found.")
        else:
            type_label = f" ({args.type})" if args.type else ""
            print(f"\nBackups{type_label}: {len(backups)} total")
            print(f"  {'Status':<7} {'ID':<10} {'Type':<12} {'Files':>8}  {'Size':>10}  {'Created':<19}  Description")
            print(f"  {'-' * 95}")
            for m in backups:
                _print_manifest(m, detailed=args.detailed)

    elif args.command == "delete":
        try:
            manifest = bm.get_backup(args.id)
            print(f"Deleting backup {_short_id(manifest.backup_id)} ({manifest.backup_type.value})...")
            bm.delete_backup(manifest.backup_id)
            print("Deleted.")
        except KeyError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    elif args.command == "prune":
        deleted = bm.prune_old(keep=args.keep)
        print(f"\nPruned {deleted} old backups (keeping {args.keep})")

    elif args.command == "stats":
        stats = bm.get_stats()
        print("\n=== Backup Statistics ===")
        print(f"  Total backups:      {stats['total_backups']}")
        print(f"  Completed:          {stats['completed']}")
        print(f"  Failed:             {stats['failed']}")
        print(f"  Max backups:        {stats['max_backups_setting']}")
        print(f"  Default compression:{stats['default_compression']}")
        print()
        print("  By type:")
        for t, count in stats["by_type"].items():
            print(f"    {t:<14} {count}")
        print()
        print("  By status:")
        for s, count in stats["by_status"].items():
            if count > 0:
                print(f"    {s:<14} {count}")
        print()
        print(f"  Total compressed:   {stats['total_compressed_size']}")
        print(f"  Total uncompressed: {stats['total_uncompressed_size']}")
        print(f"  Avg compressed:     {stats['avg_compressed_size']}")
        print(f"  Compression ratio:  {stats['compression_ratio']}")
        print(f"  Total files backed: {stats['total_file_count']}")
        print()
        if stats["oldest_backup"]:
            print(f"  Oldest backup:      {stats['oldest_backup'][:19]}")
        if stats["newest_backup"]:
            print(f"  Newest backup:      {stats['newest_backup'][:19]}")

        # Also show disk usage
        usage = bm.get_disk_usage()
        print()
        print("=== Disk Usage ===")
        print(f"  Backup directory:   {usage['backup_dir']}")
        print(f"  Total size:         {usage['total_dir_size']}")
        print(f"  Archives:           {usage['archives_size']}")
        print(f"  Snapshots:          {usage['snapshots_size']}")

    elif args.command == "diff":
        try:
            result = bm.diff_with_current(args.id)
        except KeyError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

        print(f"\n=== Diff: backup {result['backup_id']} vs current ===")
        print(f"  Backup type:     {result['backup_type']}")
        print(f"  Backup created:  {result['backup_created_at'][:19]}")
        print(f"  Files in backup: {result['total_backup_files']}")
        print(f"  Files on disk:   {result['total_current_files']}")
        print()
        print(f"  Added (new since backup):    {result['added_count']}")
        print(f"  Removed (deleted since):     {result['removed_count']}")
        print(f"  Modified (size changed):     {result['modified_count']}")
        print(f"  Unchanged:                   {result['unchanged_count']}")

        if result["added"]:
            print(f"\n  Added files ({result['added_count']}):")
            for f in result["added"][:30]:
                print(f"    + {f}")
            if result["added_count"] > 30:
                print(f"    ... and {result['added_count'] - 30} more")

        if result["removed"]:
            print(f"\n  Removed files ({result['removed_count']}):")
            for f in result["removed"][:30]:
                print(f"    - {f}")
            if result["removed_count"] > 30:
                print(f"    ... and {result['removed_count'] - 30} more")

        if result["modified"]:
            print(f"\n  Modified files ({result['modified_count']}):")
            for f in result["modified"][:30]:
                print(f"    ~ {f}")
            if result["modified_count"] > 30:
                print(f"    ... and {result['modified_count'] - 30} more")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
