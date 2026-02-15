"""
Agent Memory — Long-Term Memory Module for OpenClaw Empire Phase 5

Searchable long-term memory store for the autonomous Android phone agent.
Stores task results, app knowledge, user preferences, error logs, and
conversation context across sessions.

Capabilities:
    - Cross-session restoration and session history
    - Memory consolidation via Haiku summarization or extractive fallback
    - Tag-based indexing with inverted indices
    - Relevance scoring with time decay, priority weighting, and access bonuses
    - TTL-based automatic expiry per priority level
    - Thread-safe concurrent access with locking
    - Atomic JSON persistence to avoid data corruption
    - CLI management commands for store, recall, consolidate, prune, export, import

Data persisted to: data/memory/
    memories.json           — main memory store
    indices.json            — tag and type inverted indices
    consolidation_log.json  — consolidation history
    stats.json              — memory statistics snapshot
    archive.json            — cold storage for archived memories
    sessions.json           — session context snapshots

Usage:
    from src.agent_memory import get_memory

    memory = get_memory()

    # Store a memory
    mem = memory.store_sync(
        content="Instagram login requires 2FA code from SMS",
        type=MemoryType.APP_KNOWLEDGE,
        tags=["instagram", "login", "2fa"],
        priority=MemoryPriority.HIGH,
    )

    # Recall memories
    results = memory.recall_by_text_sync("instagram login", limit=5)

    # Build context for LLM
    context = memory.get_context_sync("how to log into instagram")

    # Consolidate old memories
    result = memory.consolidate_sync(ConsolidationStrategy.SUMMARIZE, age_days=14)
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("agent_memory")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s.%(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MEMORY_DATA_DIR = BASE_DIR / "data" / "memory"
MEMORIES_FILE = MEMORY_DATA_DIR / "memories.json"
INDICES_FILE = MEMORY_DATA_DIR / "indices.json"
CONSOLIDATION_LOG_FILE = MEMORY_DATA_DIR / "consolidation_log.json"
STATS_FILE = MEMORY_DATA_DIR / "stats.json"
ARCHIVE_FILE = MEMORY_DATA_DIR / "archive.json"
SESSIONS_FILE = MEMORY_DATA_DIR / "sessions.json"

# Ensure data directory exists on import
MEMORY_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic API
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Relevance decay
DECAY_PERIOD_DAYS = 30          # full decay cycle
MIN_RELEVANCE = 0.1             # floor for decayed relevance
DECAY_INTERVAL_SECONDS = 3600   # run decay every hour (if background enabled)

# Priority weights for scoring
PRIORITY_WEIGHTS: Dict[str, float] = {
    "critical": 2.0,
    "high": 1.5,
    "normal": 1.0,
    "low": 0.5,
    "ephemeral": 0.25,
}

# Default TTL (hours) by priority
TTL_DEFAULTS: Dict[str, Optional[int]] = {
    "critical": None,       # never expires
    "high": 90 * 24,        # 90 days
    "normal": 30 * 24,      # 30 days
    "low": 7 * 24,          # 7 days
    "ephemeral": 24,         # 24 hours
}

# Limits
MAX_MEMORIES = 50_000
MAX_CONSOLIDATION_LOG = 1000
MAX_SESSIONS = 200
MAX_EXPORT_BATCH = 10_000
CONTEXT_DEFAULT_MAX_TOKENS = 4000   # approx chars for context building
HAIKU_SUMMARY_MAX_TOKENS = 500
MERGE_SIMILARITY_THRESHOLD = 0.6

# Auto-save threshold: save after this many mutations
AUTO_SAVE_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Time helpers
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
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _hours_since(iso_str: Optional[str]) -> float:
    """Return hours elapsed since the given ISO timestamp."""
    dt = _parse_iso(iso_str)
    if dt is None:
        return 0.0
    delta = _now_utc() - dt
    return max(delta.total_seconds() / 3600.0, 0.0)


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
# Async/sync dual interface helper
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ===================================================================
# ENUMS
# ===================================================================

class MemoryType(str, Enum):
    """Categories of memory content."""
    TASK_RESULT = "task_result"
    APP_KNOWLEDGE = "app_knowledge"
    USER_PREFERENCE = "user_preference"
    ERROR_LOG = "error_log"
    CONVERSATION = "conversation"
    OBSERVATION = "observation"
    SKILL_LEARNED = "skill_learned"
    RELATIONSHIP = "relationship"
    FACT = "fact"
    PROCEDURE = "procedure"


class MemoryPriority(str, Enum):
    """Priority levels controlling TTL and relevance weighting."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    EPHEMERAL = "ephemeral"


class ConsolidationStrategy(str, Enum):
    """How to consolidate memories."""
    SUMMARIZE = "summarize"
    MERGE = "merge"
    ARCHIVE = "archive"
    DELETE = "delete"


# ===================================================================
# DATA CLASSES
# ===================================================================

@dataclass
class Memory:
    """A single memory entry in the long-term store."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = MemoryType.FACT
    content: str = ""
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MemoryPriority = MemoryPriority.NORMAL
    relevance_score: float = 1.0
    access_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    last_accessed: str = ""
    expires_at: Optional[str] = None
    source_module: str = ""
    session_id: str = ""
    linked_memories: List[str] = field(default_factory=list)
    consolidated: bool = False
    embedding_key: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        d = asdict(self)
        d["type"] = self.type.value if isinstance(self.type, MemoryType) else self.type
        d["priority"] = self.priority.value if isinstance(self.priority, MemoryPriority) else self.priority
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        """Deserialize from a plain dict loaded from JSON."""
        data = dict(data)
        if "type" in data:
            try:
                data["type"] = MemoryType(data["type"])
            except ValueError:
                data["type"] = MemoryType.FACT
        if "priority" in data:
            try:
                data["priority"] = MemoryPriority(data["priority"])
            except ValueError:
                data["priority"] = MemoryPriority.NORMAL
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def is_expired(self) -> bool:
        """Check if this memory has passed its expiry time."""
        if self.expires_at is None:
            return False
        exp = _parse_iso(self.expires_at)
        if exp is None:
            return False
        return _now_utc() > exp

    def age_hours(self) -> float:
        """Return the age of this memory in hours."""
        return _hours_since(self.created_at)


@dataclass
class MemoryQuery:
    """Parameters for searching the memory store."""
    text: str = ""
    types: List[MemoryType] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_relevance: float = 0.0
    min_priority: Optional[MemoryPriority] = None
    source_module: str = ""
    session_id: str = ""
    since: Optional[str] = None
    until: Optional[str] = None
    limit: int = 20
    include_expired: bool = False
    include_consolidated: bool = False


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    strategy: ConsolidationStrategy = ConsolidationStrategy.DELETE
    input_count: int = 0
    output_count: int = 0
    memories_affected: List[str] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategy"] = self.strategy.value if isinstance(self.strategy, ConsolidationStrategy) else self.strategy
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ConsolidationResult:
        data = dict(data)
        if "strategy" in data:
            try:
                data["strategy"] = ConsolidationStrategy(data["strategy"])
            except ValueError:
                data["strategy"] = ConsolidationStrategy.DELETE
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class MemoryStats:
    """Aggregate statistics for the memory store."""
    total_memories: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    total_accesses: int = 0
    oldest_memory: str = ""
    newest_memory: str = ""
    avg_relevance: float = 0.0
    consolidations_run: int = 0
    expired_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# TEXT SIMILARITY UTILITIES
# ===================================================================

def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _trigrams(text: str) -> Set[str]:
    """Generate character trigrams from text."""
    t = text.lower().strip()
    if len(t) < 3:
        return {t} if t else set()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def _text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two texts using combined word overlap
    and trigram Jaccard similarity.

    Returns a float between 0.0 and 1.0.
    """
    if not text1 or not text2:
        return 0.0

    # Word overlap (Jaccard)
    words1 = set(_tokenize(text1))
    words2 = set(_tokenize(text2))
    if words1 and words2:
        word_jaccard = len(words1 & words2) / len(words1 | words2)
    else:
        word_jaccard = 0.0

    # Trigram overlap (Jaccard)
    tri1 = _trigrams(text1)
    tri2 = _trigrams(text2)
    if tri1 and tri2:
        tri_jaccard = len(tri1 & tri2) / len(tri1 | tri2)
    else:
        tri_jaccard = 0.0

    # Weighted combination: words are more meaningful, trigrams catch partial matches
    return word_jaccard * 0.6 + tri_jaccard * 0.4


def _word_overlap_ratio(query_text: str, content_text: str) -> float:
    """
    Compute how many query words appear in the content.
    Returns ratio of matched query words to total query words.
    """
    if not query_text:
        return 1.0
    query_words = set(_tokenize(query_text))
    if not query_words:
        return 1.0
    content_words = set(_tokenize(content_text))
    if not content_words:
        return 0.0
    matched = query_words & content_words
    return len(matched) / len(query_words)


def _extractive_summary(texts: List[str], max_length: int = 500) -> str:
    """
    Create an extractive summary by selecting the most representative sentences.
    Used as a fallback when Haiku API is unavailable.
    """
    if not texts:
        return ""

    # Split all texts into sentences
    all_sentences: List[str] = []
    for text in texts:
        sentences = re.split(r"[.!?\n]+", text)
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                all_sentences.append(s)

    if not all_sentences:
        # Fall back to truncating concatenation
        combined = " | ".join(texts)
        return combined[:max_length]

    # Score sentences by word frequency
    word_freq: Counter = Counter()
    for sentence in all_sentences:
        for word in _tokenize(sentence):
            word_freq[word] += 1

    # Score each sentence
    scored: List[Tuple[float, int, str]] = []
    for idx, sentence in enumerate(all_sentences):
        words = _tokenize(sentence)
        if not words:
            continue
        score = sum(word_freq[w] for w in words) / len(words)
        scored.append((score, idx, sentence))

    scored.sort(key=lambda x: (-x[0], x[1]))

    # Select top sentences until max_length
    summary_parts: List[str] = []
    current_length = 0
    for _score, _idx, sentence in scored:
        if current_length + len(sentence) + 2 > max_length:
            break
        summary_parts.append(sentence)
        current_length += len(sentence) + 2

    if not summary_parts:
        return all_sentences[0][:max_length]

    return ". ".join(summary_parts) + "."


# ===================================================================
# HAIKU SUMMARIZATION (OPTIONAL)
# ===================================================================

_anthropic_available = False
_anthropic_client = None

try:
    import anthropic as _anthropic_module
    _anthropic_available = True
except ImportError:
    pass


def _get_anthropic_client():
    """Lazy-initialize the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client

    if not _anthropic_available:
        return None

    api_key = os.getenv(ANTHROPIC_API_KEY_ENV, "")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set; Haiku summarization unavailable.")
        return None

    try:
        _anthropic_client = _anthropic_module.Anthropic(api_key=api_key)
        return _anthropic_client
    except Exception as exc:
        logger.warning("Failed to initialize Anthropic client: %s", exc)
        return None


def _haiku_summarize(texts: List[str], instruction: str = "") -> Optional[str]:
    """
    Use Claude Haiku to summarize a list of text passages.
    Returns the summary string, or None if the API is unavailable or fails.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    combined = "\n---\n".join(texts)
    if not combined.strip():
        return None

    if not instruction:
        instruction = (
            "Summarize the following memory entries into a concise paragraph. "
            "Preserve key facts, relationships, and actionable knowledge. "
            "Be specific and factual."
        )

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=HAIKU_SUMMARY_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": (
                        "You are a memory consolidation engine. Summarize the provided "
                        "text entries into a compact, information-dense summary. "
                        "Do not add speculation or opinions. Preserve all facts."
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"{instruction}\n\nEntries:\n{combined}",
                }
            ],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return None
    except Exception as exc:
        logger.warning("Haiku summarization failed: %s", exc)
        return None


# ===================================================================
# PRIORITY ORDERING HELPERS
# ===================================================================

_PRIORITY_ORDER = {
    MemoryPriority.EPHEMERAL: 0,
    MemoryPriority.LOW: 1,
    MemoryPriority.NORMAL: 2,
    MemoryPriority.HIGH: 3,
    MemoryPriority.CRITICAL: 4,
}


def _priority_value(p: MemoryPriority) -> int:
    """Return an integer for priority comparison (higher = more important)."""
    return _PRIORITY_ORDER.get(p, 2)


def _priority_meets_minimum(mem_priority: MemoryPriority, min_priority: Optional[MemoryPriority]) -> bool:
    """Check if a memory's priority meets or exceeds the minimum threshold."""
    if min_priority is None:
        return True
    return _priority_value(mem_priority) >= _priority_value(min_priority)


# ===================================================================
# AGENT MEMORY
# ===================================================================

class AgentMemory:
    """
    Long-term memory store for the autonomous Android phone agent.

    Supports storage, retrieval, relevance scoring, tag indexing,
    consolidation, session management, and persistence.

    Thread-safe: all mutations are protected by a threading.Lock.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else MEMORY_DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = session_id or str(uuid.uuid4())
        self._lock = threading.Lock()
        self._dirty_count = 0

        # File paths for this instance
        self._memories_path = self._data_dir / "memories.json"
        self._indices_path = self._data_dir / "indices.json"
        self._consolidation_log_path = self._data_dir / "consolidation_log.json"
        self._stats_path = self._data_dir / "stats.json"
        self._archive_path = self._data_dir / "archive.json"
        self._sessions_path = self._data_dir / "sessions.json"

        # In-memory stores
        self._memories: Dict[str, Memory] = {}
        self._tag_index: Dict[str, List[str]] = {}   # tag -> [memory_id, ...]
        self._type_index: Dict[str, List[str]] = {}   # type -> [memory_id, ...]
        self._consolidation_log: List[dict] = []
        self._sessions: List[dict] = []

        # Load persisted state
        self._load()

        logger.info(
            "AgentMemory initialized (session=%s, memories=%d, data_dir=%s)",
            self._session_id[:8],
            len(self._memories),
            self._data_dir,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Return the current session ID."""
        return self._session_id

    @property
    def memory_count(self) -> int:
        """Return the total number of stored memories."""
        return len(self._memories)

    # ------------------------------------------------------------------
    # Persistence (private)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all stores from disk."""
        # Load memories
        raw_memories = _load_json(self._memories_path, default={})
        self._memories = {}
        if isinstance(raw_memories, dict):
            for mid, mdata in raw_memories.items():
                try:
                    mem = Memory.from_dict(mdata)
                    self._memories[mem.id] = mem
                except Exception as exc:
                    logger.warning("Failed to load memory %s: %s", mid, exc)
        elif isinstance(raw_memories, list):
            # Handle legacy list format
            for mdata in raw_memories:
                try:
                    mem = Memory.from_dict(mdata)
                    self._memories[mem.id] = mem
                except Exception as exc:
                    logger.warning("Failed to load memory from list: %s", exc)

        # Load indices
        raw_indices = _load_json(self._indices_path, default={})
        self._tag_index = raw_indices.get("tags", {})
        self._type_index = raw_indices.get("types", {})

        # Validate indices match actual memories, rebuild if needed
        if not self._validate_indices():
            logger.info("Index mismatch detected, rebuilding indices.")
            self._build_indices()

        # Load consolidation log
        raw_log = _load_json(self._consolidation_log_path, default=[])
        if isinstance(raw_log, list):
            self._consolidation_log = raw_log[-MAX_CONSOLIDATION_LOG:]
        else:
            self._consolidation_log = []

        # Load sessions
        raw_sessions = _load_json(self._sessions_path, default=[])
        if isinstance(raw_sessions, list):
            self._sessions = raw_sessions[-MAX_SESSIONS:]
        else:
            self._sessions = []

        logger.debug(
            "Loaded %d memories, %d tags, %d types, %d consolidations, %d sessions.",
            len(self._memories),
            len(self._tag_index),
            len(self._type_index),
            len(self._consolidation_log),
            len(self._sessions),
        )

    def _save(self) -> None:
        """Save all stores to disk atomically."""
        # Save memories
        memories_data = {mid: mem.to_dict() for mid, mem in self._memories.items()}
        _save_json(self._memories_path, memories_data)

        # Save indices
        self._save_indices()

        # Save consolidation log
        _save_json(
            self._consolidation_log_path,
            self._consolidation_log[-MAX_CONSOLIDATION_LOG:],
        )

        # Save sessions
        _save_json(self._sessions_path, self._sessions[-MAX_SESSIONS:])

        self._dirty_count = 0
        logger.debug("Memory state saved to disk.")

    def _save_indices(self) -> None:
        """Save tag and type indices to disk."""
        indices_data = {
            "tags": self._tag_index,
            "types": self._type_index,
        }
        _save_json(self._indices_path, indices_data)

    def _auto_save(self) -> None:
        """Increment dirty counter and save if threshold reached."""
        self._dirty_count += 1
        if self._dirty_count >= AUTO_SAVE_THRESHOLD:
            try:
                self._save()
            except Exception as exc:
                logger.error("Auto-save failed: %s", exc)

    def _force_save(self) -> None:
        """Force an immediate save regardless of dirty count."""
        try:
            self._save()
        except Exception as exc:
            logger.error("Force save failed: %s", exc)

    # ------------------------------------------------------------------
    # Index management (private)
    # ------------------------------------------------------------------

    def _validate_indices(self) -> bool:
        """Quick validation that indices are consistent with memories."""
        # Check that all memory IDs in indices exist in memories
        all_tag_ids: Set[str] = set()
        for ids in self._tag_index.values():
            all_tag_ids.update(ids)

        all_type_ids: Set[str] = set()
        for ids in self._type_index.values():
            all_type_ids.update(ids)

        memory_ids = set(self._memories.keys())

        # If any indexed ID is not in memories, indices are stale
        if all_tag_ids - memory_ids:
            return False
        if all_type_ids - memory_ids:
            return False

        # If memories exist but indices are empty, rebuild needed
        if self._memories and not self._tag_index and not self._type_index:
            return False

        return True

    def _build_indices(self) -> None:
        """Rebuild all indices from the memory store."""
        self._tag_index = {}
        self._type_index = {}

        for mid, mem in self._memories.items():
            self._add_to_index(mem)

        logger.debug(
            "Rebuilt indices: %d tags, %d types.",
            len(self._tag_index),
            len(self._type_index),
        )

    def _add_to_index(self, memory: Memory) -> None:
        """Add a single memory to both tag and type indices."""
        # Type index
        type_key = memory.type.value if isinstance(memory.type, MemoryType) else str(memory.type)
        if type_key not in self._type_index:
            self._type_index[type_key] = []
        if memory.id not in self._type_index[type_key]:
            self._type_index[type_key].append(memory.id)

        # Tag index
        for tag in memory.tags:
            tag_lower = tag.lower().strip()
            if not tag_lower:
                continue
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = []
            if memory.id not in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].append(memory.id)

    def _remove_from_index(self, memory_id: str) -> None:
        """Remove a memory from all indices."""
        # Remove from tag index
        empty_tags: List[str] = []
        for tag, ids in self._tag_index.items():
            if memory_id in ids:
                ids.remove(memory_id)
            if not ids:
                empty_tags.append(tag)
        for tag in empty_tags:
            del self._tag_index[tag]

        # Remove from type index
        empty_types: List[str] = []
        for type_key, ids in self._type_index.items():
            if memory_id in ids:
                ids.remove(memory_id)
            if not ids:
                empty_types.append(type_key)
        for type_key in empty_types:
            del self._type_index[type_key]

    def _update_index_tags(self, memory: Memory, old_tags: List[str]) -> None:
        """Update tag index when a memory's tags change."""
        old_set = {t.lower().strip() for t in old_tags if t.strip()}
        new_set = {t.lower().strip() for t in memory.tags if t.strip()}

        # Remove from tags no longer present
        removed_tags = old_set - new_set
        for tag in removed_tags:
            if tag in self._tag_index and memory.id in self._tag_index[tag]:
                self._tag_index[tag].remove(memory.id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

        # Add to new tags
        added_tags = new_set - old_set
        for tag in added_tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            if memory.id not in self._tag_index[tag]:
                self._tag_index[tag].append(memory.id)

    # ------------------------------------------------------------------
    # Relevance & Scoring (private)
    # ------------------------------------------------------------------

    def _calculate_relevance(self, memory: Memory, query: MemoryQuery) -> float:
        """
        Calculate a composite relevance score for a memory given a query.

        Score formula:
            score = base_relevance * priority_weight * recency_factor * access_bonus * text_match

        Components:
            - base_relevance: the memory's stored relevance_score
            - priority_weight: CRITICAL=2.0, HIGH=1.5, NORMAL=1.0, LOW=0.5, EPHEMERAL=0.25
            - recency_factor: 1.0 - (age_hours / (24 * 30)), clamped to [MIN_RELEVANCE, 1.0]
            - access_bonus: 1.0 + log(access_count + 1) * 0.1
            - text_match: word overlap ratio between query text and memory content, 1.0 if no query
        """
        # Base relevance
        base = max(memory.relevance_score, MIN_RELEVANCE)

        # Priority weight
        priority_key = memory.priority.value if isinstance(memory.priority, MemoryPriority) else str(memory.priority)
        p_weight = PRIORITY_WEIGHTS.get(priority_key, 1.0)

        # Recency factor: decay over DECAY_PERIOD_DAYS days
        age_h = memory.age_hours()
        decay_hours = DECAY_PERIOD_DAYS * 24.0
        recency = 1.0 - (age_h / decay_hours)
        recency = max(recency, MIN_RELEVANCE)
        recency = min(recency, 1.0)

        # Access bonus: logarithmic boost
        access_bonus = 1.0 + math.log(memory.access_count + 1) * 0.1

        # Text match
        if query.text and query.text.strip():
            # Match against content and summary
            content_match = _word_overlap_ratio(query.text, memory.content)
            summary_match = _word_overlap_ratio(query.text, memory.summary) if memory.summary else 0.0
            tag_text = " ".join(memory.tags)
            tag_match = _word_overlap_ratio(query.text, tag_text) if tag_text else 0.0
            # Best of content, summary, and tags
            text_match = max(content_match, summary_match * 0.8, tag_match * 0.6)
        else:
            text_match = 1.0

        score = base * p_weight * recency * access_bonus * text_match
        return round(score, 6)

    def _decay_relevance(self) -> int:
        """
        Apply time-based decay to all memory relevance scores.
        Called periodically. Returns the number of memories affected.
        """
        affected = 0
        now = _now_utc()

        with self._lock:
            for mem in self._memories.values():
                created = _parse_iso(mem.created_at)
                if created is None:
                    continue

                age_h = (now - created).total_seconds() / 3600.0
                decay_hours = DECAY_PERIOD_DAYS * 24.0

                # Calculate new base relevance
                new_relevance = 1.0 - (age_h / decay_hours)
                new_relevance = max(new_relevance, MIN_RELEVANCE)
                new_relevance = min(new_relevance, 1.0)

                # Critical memories decay slower
                priority_key = mem.priority.value if isinstance(mem.priority, MemoryPriority) else str(mem.priority)
                if priority_key == "critical":
                    new_relevance = max(new_relevance, 0.8)
                elif priority_key == "high":
                    new_relevance = max(new_relevance, 0.4)

                if abs(mem.relevance_score - new_relevance) > 0.001:
                    mem.relevance_score = round(new_relevance, 6)
                    affected += 1

            if affected > 0:
                self._auto_save()

        return affected

    def _boost_relevance(self, memory_id: str, amount: float = 0.05) -> None:
        """Boost a memory's relevance score (called on access)."""
        mem = self._memories.get(memory_id)
        if mem is None:
            return
        mem.relevance_score = min(mem.relevance_score + amount, 2.0)

    # ------------------------------------------------------------------
    # Storage (async)
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        type: MemoryType = MemoryType.FACT,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        source_module: str = "",
        ttl_hours: Optional[int] = None,
        linked_memories: Optional[List[str]] = None,
        summary: str = "",
    ) -> Memory:
        """
        Create and persist a new memory.

        Args:
            content:         Main text content of the memory.
            type:            Category of this memory (MemoryType enum).
            tags:            List of tags for indexing and retrieval.
            metadata:        Arbitrary metadata dict.
            priority:        Importance level (affects TTL and relevance).
            source_module:   Which module created this memory.
            ttl_hours:       Custom TTL in hours. None uses priority default.
            linked_memories: List of related memory IDs.
            summary:         Optional pre-computed summary.

        Returns:
            The created Memory object.
        """
        # Compute expiry
        if ttl_hours is not None:
            expires_at = (_now_utc() + timedelta(hours=ttl_hours)).isoformat()
        else:
            priority_key = priority.value if isinstance(priority, MemoryPriority) else str(priority)
            default_ttl = TTL_DEFAULTS.get(priority_key)
            if default_ttl is not None:
                expires_at = (_now_utc() + timedelta(hours=default_ttl)).isoformat()
            else:
                expires_at = None

        mem = Memory(
            id=str(uuid.uuid4()),
            type=type,
            content=content,
            summary=summary,
            tags=tags or [],
            metadata=metadata or {},
            priority=priority,
            relevance_score=1.0,
            access_count=0,
            created_at=_now_iso(),
            last_accessed="",
            expires_at=expires_at,
            source_module=source_module,
            session_id=self._session_id,
            linked_memories=linked_memories or [],
            consolidated=False,
            embedding_key="",
        )

        with self._lock:
            # Enforce maximum memories
            if len(self._memories) >= MAX_MEMORIES:
                self._evict_lowest_relevance(count=1)

            self._memories[mem.id] = mem
            self._add_to_index(mem)
            self._auto_save()

        logger.info(
            "Stored memory %s (type=%s, priority=%s, tags=%s)",
            mem.id[:8],
            type.value,
            priority.value,
            ",".join(mem.tags[:5]),
        )
        return mem

    def store_sync(
        self,
        content: str,
        type: MemoryType = MemoryType.FACT,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        source_module: str = "",
        ttl_hours: Optional[int] = None,
        linked_memories: Optional[List[str]] = None,
        summary: str = "",
    ) -> Memory:
        """Synchronous wrapper for store()."""
        return _run_sync(self.store(
            content=content, type=type, tags=tags, metadata=metadata,
            priority=priority, source_module=source_module, ttl_hours=ttl_hours,
            linked_memories=linked_memories, summary=summary,
        ))

    async def store_batch(self, memories: List[dict]) -> List[Memory]:
        """
        Bulk store multiple memories at once.

        Each dict in the list should contain kwargs for store():
            content (required), type, tags, metadata, priority, source_module,
            ttl_hours, linked_memories, summary.

        Returns a list of created Memory objects.
        """
        results: List[Memory] = []
        for mem_dict in memories:
            if "content" not in mem_dict:
                logger.warning("Skipping memory without content field.")
                continue

            # Parse enums if provided as strings
            if "type" in mem_dict and isinstance(mem_dict["type"], str):
                try:
                    mem_dict["type"] = MemoryType(mem_dict["type"])
                except ValueError:
                    mem_dict["type"] = MemoryType.FACT

            if "priority" in mem_dict and isinstance(mem_dict["priority"], str):
                try:
                    mem_dict["priority"] = MemoryPriority(mem_dict["priority"])
                except ValueError:
                    mem_dict["priority"] = MemoryPriority.NORMAL

            mem = await self.store(**mem_dict)
            results.append(mem)

        logger.info("Batch stored %d memories.", len(results))
        return results

    def store_batch_sync(self, memories: List[dict]) -> List[Memory]:
        """Synchronous wrapper for store_batch()."""
        return _run_sync(self.store_batch(memories))

    async def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[MemoryPriority] = None,
        summary: Optional[str] = None,
        linked_memories: Optional[List[str]] = None,
    ) -> Memory:
        """
        Update an existing memory's fields.

        Only non-None arguments are applied. Tags update triggers index rebuild
        for that memory. Returns the updated Memory.

        Raises KeyError if the memory is not found.
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                raise KeyError(f"Memory {memory_id} not found.")

            old_tags = list(mem.tags)

            if content is not None:
                mem.content = content
            if tags is not None:
                mem.tags = tags
            if metadata is not None:
                mem.metadata = metadata
            if priority is not None:
                mem.priority = priority
                # Recalculate expiry based on new priority
                priority_key = priority.value if isinstance(priority, MemoryPriority) else str(priority)
                default_ttl = TTL_DEFAULTS.get(priority_key)
                if default_ttl is not None:
                    mem.expires_at = (_now_utc() + timedelta(hours=default_ttl)).isoformat()
                else:
                    mem.expires_at = None
            if summary is not None:
                mem.summary = summary
            if linked_memories is not None:
                mem.linked_memories = linked_memories

            # Update indices if tags changed
            if tags is not None:
                self._update_index_tags(mem, old_tags)

            # Update type index if needed (type is immutable, but ensure consistent)
            self._auto_save()

        logger.info("Updated memory %s.", memory_id[:8])
        return mem

    def update_sync(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[MemoryPriority] = None,
        summary: Optional[str] = None,
        linked_memories: Optional[List[str]] = None,
    ) -> Memory:
        """Synchronous wrapper for update()."""
        return _run_sync(self.update(
            memory_id=memory_id, content=content, tags=tags,
            metadata=metadata, priority=priority, summary=summary,
            linked_memories=linked_memories,
        ))

    async def delete(self, memory_id: str) -> bool:
        """
        Soft-delete a memory by setting its expiry to now.
        The memory remains in the store until the next prune/expire cycle.
        Returns True if the memory was found and marked.
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                logger.warning("Cannot delete memory %s: not found.", memory_id[:8])
                return False

            mem.expires_at = _now_iso()
            self._auto_save()

        logger.info("Soft-deleted memory %s.", memory_id[:8])
        return True

    def delete_sync(self, memory_id: str) -> bool:
        """Synchronous wrapper for delete()."""
        return _run_sync(self.delete(memory_id))

    async def hard_delete(self, memory_id: str) -> bool:
        """
        Permanently remove a memory from the store and all indices.
        Returns True if the memory was found and removed.
        """
        with self._lock:
            mem = self._memories.get(memory_id)
            if mem is None:
                logger.warning("Cannot hard-delete memory %s: not found.", memory_id[:8])
                return False

            self._remove_from_index(memory_id)
            del self._memories[memory_id]
            self._auto_save()

        logger.info("Hard-deleted memory %s.", memory_id[:8])
        return True

    def hard_delete_sync(self, memory_id: str) -> bool:
        """Synchronous wrapper for hard_delete()."""
        return _run_sync(self.hard_delete(memory_id))

    def _evict_lowest_relevance(self, count: int = 1) -> int:
        """
        Remove the N lowest-relevance non-critical memories to make room.
        Must be called while self._lock is held.
        Returns the number actually evicted.
        """
        # Sort by relevance ascending, skip CRITICAL
        candidates = [
            (mid, mem)
            for mid, mem in self._memories.items()
            if mem.priority != MemoryPriority.CRITICAL
        ]
        candidates.sort(key=lambda x: x[1].relevance_score)

        evicted = 0
        for mid, mem in candidates[:count]:
            self._remove_from_index(mid)
            del self._memories[mid]
            evicted += 1
            logger.debug("Evicted memory %s (relevance=%.4f).", mid[:8], mem.relevance_score)

        return evicted

    # ------------------------------------------------------------------
    # Retrieval (async)
    # ------------------------------------------------------------------

    async def recall(self, query: MemoryQuery) -> List[Memory]:
        """
        Main search method. Finds memories matching the query, scores them
        by relevance, and returns the top results.

        Updates access_count and last_accessed on returned memories.
        """
        with self._lock:
            candidates = self._filter_candidates(query)

            # Score and sort
            scored: List[Tuple[float, Memory]] = []
            for mem in candidates:
                score = self._calculate_relevance(mem, query)
                if score >= query.min_relevance:
                    scored.append((score, mem))

            scored.sort(key=lambda x: -x[0])
            results = [mem for _score, mem in scored[:query.limit]]

            # Update access tracking
            now = _now_iso()
            for mem in results:
                mem.access_count += 1
                mem.last_accessed = now
                self._boost_relevance(mem.id, 0.02)

            if results:
                self._auto_save()

        return results

    def recall_sync(self, query: MemoryQuery) -> List[Memory]:
        """Synchronous wrapper for recall()."""
        return _run_sync(self.recall(query))

    def _filter_candidates(self, query: MemoryQuery) -> List[Memory]:
        """
        Filter memories based on query constraints (type, tags, expiry, etc.).
        Must be called while self._lock is held.
        Returns a list of candidate Memory objects.
        """
        # Start with all memories or narrow by type/tag indices
        candidate_ids: Optional[Set[str]] = None

        # Type filter via index
        if query.types:
            type_ids: Set[str] = set()
            for t in query.types:
                type_key = t.value if isinstance(t, MemoryType) else str(t)
                ids = self._type_index.get(type_key, [])
                type_ids.update(ids)
            candidate_ids = type_ids

        # Tag filter via index
        if query.tags:
            for tag in query.tags:
                tag_lower = tag.lower().strip()
                tag_ids = set(self._tag_index.get(tag_lower, []))
                if candidate_ids is None:
                    candidate_ids = tag_ids
                else:
                    # Intersect: memory must match all tags AND types
                    candidate_ids = candidate_ids & tag_ids

        # Resolve IDs to Memory objects
        if candidate_ids is not None:
            candidates = [
                self._memories[mid]
                for mid in candidate_ids
                if mid in self._memories
            ]
        else:
            candidates = list(self._memories.values())

        # Apply remaining filters
        filtered: List[Memory] = []
        for mem in candidates:
            # Expiry filter
            if not query.include_expired and mem.is_expired():
                continue

            # Consolidated filter
            if not query.include_consolidated and mem.consolidated:
                continue

            # Priority filter
            if not _priority_meets_minimum(mem.priority, query.min_priority):
                continue

            # Source module filter
            if query.source_module and mem.source_module != query.source_module:
                continue

            # Session filter
            if query.session_id and mem.session_id != query.session_id:
                continue

            # Time range filters
            if query.since:
                since_dt = _parse_iso(query.since)
                created_dt = _parse_iso(mem.created_at)
                if since_dt and created_dt and created_dt < since_dt:
                    continue

            if query.until:
                until_dt = _parse_iso(query.until)
                created_dt = _parse_iso(mem.created_at)
                if until_dt and created_dt and created_dt > until_dt:
                    continue

            filtered.append(mem)

        return filtered

    async def recall_by_text(
        self,
        text: str,
        limit: int = 20,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Memory]:
        """
        Convenience method: search memories by text content.
        """
        query = MemoryQuery(
            text=text,
            types=types or [],
            limit=limit,
        )
        return await self.recall(query)

    def recall_by_text_sync(
        self,
        text: str,
        limit: int = 20,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Memory]:
        """Synchronous wrapper for recall_by_text()."""
        return _run_sync(self.recall_by_text(text, limit, types))

    async def recall_by_tags(
        self,
        tags: List[str],
        match_all: bool = True,
        limit: int = 20,
    ) -> List[Memory]:
        """
        Recall memories by tags.

        Args:
            tags:       List of tag strings to search for.
            match_all:  If True, memory must have ALL tags. If False, ANY tag matches.
            limit:      Max results.
        """
        if match_all:
            # Use query which does intersection
            query = MemoryQuery(tags=tags, limit=limit)
            return await self.recall(query)
        else:
            # Union: find memories matching ANY tag
            with self._lock:
                candidate_ids: Set[str] = set()
                for tag in tags:
                    tag_lower = tag.lower().strip()
                    ids = self._tag_index.get(tag_lower, [])
                    candidate_ids.update(ids)

                candidates: List[Memory] = []
                for mid in candidate_ids:
                    mem = self._memories.get(mid)
                    if mem is None:
                        continue
                    if mem.is_expired():
                        continue
                    if mem.consolidated:
                        continue
                    candidates.append(mem)

                # Score by number of matching tags
                def tag_match_count(m: Memory) -> int:
                    m_tags = {t.lower().strip() for t in m.tags}
                    q_tags = {t.lower().strip() for t in tags}
                    return len(m_tags & q_tags)

                candidates.sort(key=lambda m: (-tag_match_count(m), -m.relevance_score))
                results = candidates[:limit]

                # Update access tracking
                now = _now_iso()
                for mem in results:
                    mem.access_count += 1
                    mem.last_accessed = now

                if results:
                    self._auto_save()

            return results

    def recall_by_tags_sync(
        self,
        tags: List[str],
        match_all: bool = True,
        limit: int = 20,
    ) -> List[Memory]:
        """Synchronous wrapper for recall_by_tags()."""
        return _run_sync(self.recall_by_tags(tags, match_all, limit))

    async def recall_recent(
        self,
        limit: int = 20,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Memory]:
        """Retrieve the most recently created memories."""
        with self._lock:
            candidates = list(self._memories.values())

            # Filter by type
            if types:
                type_values = {t.value for t in types}
                candidates = [
                    m for m in candidates
                    if (m.type.value if isinstance(m.type, MemoryType) else str(m.type)) in type_values
                ]

            # Filter out expired and consolidated
            candidates = [
                m for m in candidates
                if not m.is_expired() and not m.consolidated
            ]

            # Sort by created_at descending
            candidates.sort(key=lambda m: m.created_at or "", reverse=True)
            results = candidates[:limit]

            # Update access tracking
            now = _now_iso()
            for mem in results:
                mem.access_count += 1
                mem.last_accessed = now

            if results:
                self._auto_save()

        return results

    def recall_recent_sync(
        self,
        limit: int = 20,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Memory]:
        """Synchronous wrapper for recall_recent()."""
        return _run_sync(self.recall_recent(limit, types))

    async def recall_frequent(
        self,
        limit: int = 20,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Memory]:
        """Retrieve the most frequently accessed memories."""
        with self._lock:
            candidates = list(self._memories.values())

            # Filter by type
            if types:
                type_values = {t.value for t in types}
                candidates = [
                    m for m in candidates
                    if (m.type.value if isinstance(m.type, MemoryType) else str(m.type)) in type_values
                ]

            # Filter out expired and consolidated
            candidates = [
                m for m in candidates
                if not m.is_expired() and not m.consolidated
            ]

            # Sort by access_count descending, then recency
            candidates.sort(key=lambda m: (-m.access_count, m.created_at or ""), reverse=False)
            results = candidates[:limit]

            # Update access tracking
            now = _now_iso()
            for mem in results:
                mem.access_count += 1
                mem.last_accessed = now

            if results:
                self._auto_save()

        return results

    def recall_frequent_sync(
        self,
        limit: int = 20,
        types: Optional[List[MemoryType]] = None,
    ) -> List[Memory]:
        """Synchronous wrapper for recall_frequent()."""
        return _run_sync(self.recall_frequent(limit, types))

    async def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a single memory by ID.
        Returns None if not found. Does NOT update access tracking.
        """
        with self._lock:
            return self._memories.get(memory_id)

    def get_sync(self, memory_id: str) -> Optional[Memory]:
        """Synchronous wrapper for get()."""
        return _run_sync(self.get(memory_id))

    async def get_context(
        self,
        topic: str,
        max_tokens: int = CONTEXT_DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Build a context string from relevant memories for inclusion in LLM prompts.

        Searches for memories related to the topic, formats them as a structured
        text block, and truncates to approximately max_tokens characters.

        Args:
            topic:      The topic or question to find relevant context for.
            max_tokens: Approximate maximum character count for the returned string.

        Returns:
            A formatted string of relevant memory context.
        """
        query = MemoryQuery(text=topic, limit=30)
        results = await self.recall(query)

        if not results:
            return ""

        lines: List[str] = []
        current_length = 0
        header = f"=== Agent Memory Context for: {topic} ===\n"
        lines.append(header)
        current_length += len(header)

        for i, mem in enumerate(results):
            # Format each memory entry
            type_label = mem.type.value if isinstance(mem.type, MemoryType) else str(mem.type)
            priority_label = mem.priority.value if isinstance(mem.priority, MemoryPriority) else str(mem.priority)
            tags_str = ", ".join(mem.tags[:5]) if mem.tags else "none"

            entry_header = f"\n[{i + 1}] {type_label} (priority={priority_label}, tags={tags_str})"

            # Use summary if available, otherwise content
            display_text = mem.summary if mem.summary else mem.content
            # Truncate individual memory to reasonable size
            if len(display_text) > 500:
                display_text = display_text[:497] + "..."

            entry = f"{entry_header}\n{display_text}\n"
            entry_length = len(entry)

            if current_length + entry_length > max_tokens:
                # Check if we can fit a truncated version
                remaining = max_tokens - current_length - len(entry_header) - 10
                if remaining > 50:
                    truncated = display_text[:remaining] + "..."
                    entry = f"{entry_header}\n{truncated}\n"
                    lines.append(entry)
                break

            lines.append(entry)
            current_length += entry_length

        lines.append("\n=== End Memory Context ===")
        return "".join(lines)

    def get_context_sync(
        self,
        topic: str,
        max_tokens: int = CONTEXT_DEFAULT_MAX_TOKENS,
    ) -> str:
        """Synchronous wrapper for get_context()."""
        return _run_sync(self.get_context(topic, max_tokens))

    # ------------------------------------------------------------------
    # Consolidation (async)
    # ------------------------------------------------------------------

    async def consolidate(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.SUMMARIZE,
        type_filter: Optional[MemoryType] = None,
        age_days: int = 14,
        max_batch: int = 50,
    ) -> ConsolidationResult:
        """
        Run a consolidation operation on memories.

        Args:
            strategy:    How to consolidate (summarize, merge, archive, delete).
            type_filter: Only consolidate memories of this type (None = all).
            age_days:    Only consolidate memories older than this many days.
            max_batch:   Maximum number of memories to consolidate in one run.

        Returns:
            ConsolidationResult with details of what was done.
        """
        logger.info(
            "Starting consolidation: strategy=%s, type=%s, age_days=%d, max_batch=%d",
            strategy.value,
            type_filter.value if type_filter else "all",
            age_days,
            max_batch,
        )

        if strategy == ConsolidationStrategy.SUMMARIZE:
            result = await self._consolidate_summarize(type_filter, age_days, max_batch)
        elif strategy == ConsolidationStrategy.MERGE:
            result = await self._consolidate_merge(type_filter, age_days, max_batch)
        elif strategy == ConsolidationStrategy.ARCHIVE:
            result = await self._consolidate_archive(age_days, max_batch)
        elif strategy == ConsolidationStrategy.DELETE:
            result = await self._consolidate_delete()
        else:
            result = ConsolidationResult(
                strategy=strategy,
                summary=f"Unknown strategy: {strategy}",
            )

        # Log the consolidation
        with self._lock:
            self._consolidation_log.append(result.to_dict())
            if len(self._consolidation_log) > MAX_CONSOLIDATION_LOG:
                self._consolidation_log = self._consolidation_log[-MAX_CONSOLIDATION_LOG:]
            self._force_save()

        logger.info(
            "Consolidation complete: strategy=%s, input=%d, output=%d",
            result.strategy.value,
            result.input_count,
            result.output_count,
        )
        return result

    def consolidate_sync(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.SUMMARIZE,
        type_filter: Optional[MemoryType] = None,
        age_days: int = 14,
        max_batch: int = 50,
    ) -> ConsolidationResult:
        """Synchronous wrapper for consolidate()."""
        return _run_sync(self.consolidate(strategy, type_filter, age_days, max_batch))

    async def _consolidate_summarize(
        self,
        type_filter: Optional[MemoryType],
        age_days: int,
        max_batch: int,
    ) -> ConsolidationResult:
        """
        Summarize old memories: generate a summary for each, mark as consolidated.
        Uses Haiku for summarization with extractive fallback.
        """
        cutoff = (_now_utc() - timedelta(days=age_days)).isoformat()
        affected_ids: List[str] = []

        with self._lock:
            # Find eligible memories
            candidates: List[Memory] = []
            for mem in self._memories.values():
                if mem.consolidated:
                    continue
                if mem.is_expired():
                    continue
                if mem.priority == MemoryPriority.CRITICAL:
                    continue
                if type_filter is not None and mem.type != type_filter:
                    continue
                if mem.created_at > cutoff:
                    continue
                candidates.append(mem)

            candidates.sort(key=lambda m: m.created_at or "")
            batch = candidates[:max_batch]

        if not batch:
            return ConsolidationResult(
                strategy=ConsolidationStrategy.SUMMARIZE,
                input_count=0,
                output_count=0,
                summary="No eligible memories found for summarization.",
            )

        # Group by type for better summaries
        by_type: Dict[str, List[Memory]] = {}
        for mem in batch:
            type_key = mem.type.value if isinstance(mem.type, MemoryType) else str(mem.type)
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(mem)

        summaries_generated = 0
        for type_key, type_batch in by_type.items():
            # Process in sub-batches of 10 for summarization
            for i in range(0, len(type_batch), 10):
                sub_batch = type_batch[i:i + 10]
                texts = [m.content for m in sub_batch]

                # Try Haiku summarization first
                summary = _haiku_summarize(
                    texts,
                    instruction=(
                        f"Summarize these {len(texts)} {type_key} memory entries. "
                        "Preserve all key facts and actionable information."
                    ),
                )

                # Fall back to extractive summary
                if summary is None:
                    summary = _extractive_summary(texts, max_length=500)

                # Apply summary to each memory in the sub-batch
                with self._lock:
                    for mem in sub_batch:
                        if len(sub_batch) == 1:
                            # Single memory: just summarize its content
                            mem.summary = summary
                        else:
                            # Group summary: each gets the group summary reference
                            mem.summary = f"[Consolidated with {len(sub_batch)} entries] {summary}"
                        mem.consolidated = True
                        affected_ids.append(mem.id)
                        summaries_generated += 1

        return ConsolidationResult(
            strategy=ConsolidationStrategy.SUMMARIZE,
            input_count=len(batch),
            output_count=summaries_generated,
            memories_affected=affected_ids,
            summary=f"Summarized {summaries_generated} memories across {len(by_type)} types.",
        )

    async def _consolidate_merge(
        self,
        type_filter: Optional[MemoryType],
        age_days: int,
        max_batch: int,
    ) -> ConsolidationResult:
        """
        Merge similar memories: find pairs with high text overlap,
        combine them into one, and mark the duplicate as consolidated.
        """
        cutoff = (_now_utc() - timedelta(days=age_days)).isoformat()
        affected_ids: List[str] = []
        merge_count = 0

        with self._lock:
            # Find eligible memories
            candidates: List[Memory] = []
            for mem in self._memories.values():
                if mem.consolidated:
                    continue
                if mem.is_expired():
                    continue
                if mem.priority == MemoryPriority.CRITICAL:
                    continue
                if type_filter is not None and mem.type != type_filter:
                    continue
                if mem.created_at > cutoff:
                    continue
                candidates.append(mem)

            candidates.sort(key=lambda m: m.created_at or "")
            batch = candidates[:max_batch]

            # Find similar pairs
            merged_ids: Set[str] = set()
            for i in range(len(batch)):
                if batch[i].id in merged_ids:
                    continue
                for j in range(i + 1, len(batch)):
                    if batch[j].id in merged_ids:
                        continue

                    sim = _text_similarity(batch[i].content, batch[j].content)
                    if sim >= MERGE_SIMILARITY_THRESHOLD:
                        # Merge j into i
                        primary = batch[i]
                        secondary = batch[j]

                        # Combine content
                        if len(primary.content) < len(secondary.content):
                            primary.content = secondary.content
                        # Merge tags
                        all_tags = list(set(primary.tags + secondary.tags))
                        old_tags = list(primary.tags)
                        primary.tags = all_tags
                        # Merge metadata
                        merged_meta = dict(secondary.metadata)
                        merged_meta.update(primary.metadata)
                        primary.metadata = merged_meta
                        # Link
                        if secondary.id not in primary.linked_memories:
                            primary.linked_memories.append(secondary.id)
                        # Take higher priority
                        if _priority_value(secondary.priority) > _priority_value(primary.priority):
                            primary.priority = secondary.priority
                        # Combine access counts
                        primary.access_count += secondary.access_count

                        # Mark secondary as consolidated
                        secondary.consolidated = True
                        secondary.summary = f"Merged into {primary.id[:8]}"

                        # Update indices for primary
                        self._update_index_tags(primary, old_tags)

                        merged_ids.add(secondary.id)
                        affected_ids.append(secondary.id)
                        affected_ids.append(primary.id)
                        merge_count += 1

        return ConsolidationResult(
            strategy=ConsolidationStrategy.MERGE,
            input_count=len(batch),
            output_count=merge_count,
            memories_affected=affected_ids,
            summary=f"Merged {merge_count} similar memory pairs from {len(batch)} candidates.",
        )

    async def _consolidate_archive(self, age_days: int, max_batch: int) -> ConsolidationResult:
        """
        Move old, low-priority memories to an archive file.
        """
        cutoff = (_now_utc() - timedelta(days=age_days)).isoformat()
        archived_ids: List[str] = []

        with self._lock:
            # Load existing archive
            archive = _load_json(self._archive_path, default=[])
            if not isinstance(archive, list):
                archive = []

            # Find eligible memories (LOW and EPHEMERAL only)
            candidates: List[Memory] = []
            for mem in self._memories.values():
                if mem.consolidated:
                    continue
                if mem.is_expired():
                    continue
                if mem.priority not in (MemoryPriority.LOW, MemoryPriority.EPHEMERAL):
                    continue
                if mem.created_at > cutoff:
                    continue
                candidates.append(mem)

            candidates.sort(key=lambda m: m.relevance_score)
            batch = candidates[:max_batch]

            # Archive and remove
            for mem in batch:
                archive.append(mem.to_dict())
                self._remove_from_index(mem.id)
                del self._memories[mem.id]
                archived_ids.append(mem.id)

            # Save archive
            _save_json(self._archive_path, archive)

        return ConsolidationResult(
            strategy=ConsolidationStrategy.ARCHIVE,
            input_count=len(batch),
            output_count=len(archived_ids),
            memories_affected=archived_ids,
            summary=f"Archived {len(archived_ids)} low-priority memories older than {age_days} days.",
        )

    async def _consolidate_delete(self) -> ConsolidationResult:
        """
        Remove all expired memories from the store.
        """
        expired_count = await self._expire_old()

        return ConsolidationResult(
            strategy=ConsolidationStrategy.DELETE,
            input_count=expired_count,
            output_count=0,
            memories_affected=[],
            summary=f"Deleted {expired_count} expired memories.",
        )

    async def _summarize_memories(self, memories: List[Memory]) -> str:
        """
        Generate a summary for a group of memories.
        Uses Haiku first, falls back to extractive summarization.
        """
        texts = [m.content for m in memories]

        # Try Haiku
        haiku_summary = _haiku_summarize(texts)
        if haiku_summary:
            return haiku_summary

        # Extractive fallback
        return _extractive_summary(texts, max_length=500)

    async def _merge_similar(
        self,
        memories: List[Memory],
        similarity_threshold: float = MERGE_SIMILARITY_THRESHOLD,
    ) -> List[Memory]:
        """
        Find and merge memories with content similarity above the threshold.
        Returns the list of merged (surviving) memories.
        """
        if not memories:
            return []

        merged_ids: Set[str] = set()
        surviving: List[Memory] = []

        for i, mem_i in enumerate(memories):
            if mem_i.id in merged_ids:
                continue

            # Find all similar memories
            cluster = [mem_i]
            for j in range(i + 1, len(memories)):
                mem_j = memories[j]
                if mem_j.id in merged_ids:
                    continue
                sim = _text_similarity(mem_i.content, mem_j.content)
                if sim >= similarity_threshold:
                    cluster.append(mem_j)
                    merged_ids.add(mem_j.id)

            if len(cluster) > 1:
                # Merge cluster into primary
                primary = cluster[0]
                for secondary in cluster[1:]:
                    # Keep longer content
                    if len(secondary.content) > len(primary.content):
                        primary.content = secondary.content
                    # Merge tags
                    primary.tags = list(set(primary.tags + secondary.tags))
                    # Merge metadata
                    for k, v in secondary.metadata.items():
                        if k not in primary.metadata:
                            primary.metadata[k] = v
                    # Link
                    primary.linked_memories.append(secondary.id)
                    primary.access_count += secondary.access_count

                    # Mark secondary
                    with self._lock:
                        if secondary.id in self._memories:
                            self._memories[secondary.id].consolidated = True
                            self._memories[secondary.id].summary = f"Merged into {primary.id[:8]}"

            surviving.append(mem_i)

        return surviving

    async def _archive_old(self, age_days: int = 30) -> int:
        """
        Move old low-priority memories to the archive file.
        Returns the number of memories archived.
        """
        result = await self._consolidate_archive(age_days, max_batch=100)
        return result.output_count

    async def _expire_old(self) -> int:
        """
        Remove all memories that have passed their TTL.
        Returns the number of memories removed.
        """
        expired_ids: List[str] = []

        with self._lock:
            for mid, mem in list(self._memories.items()):
                if mem.is_expired():
                    expired_ids.append(mid)

            for mid in expired_ids:
                self._remove_from_index(mid)
                del self._memories[mid]

            if expired_ids:
                self._force_save()

        if expired_ids:
            logger.info("Expired %d memories.", len(expired_ids))

        return len(expired_ids)

    def _expire_old_sync(self) -> int:
        """Synchronous wrapper for _expire_old()."""
        return _run_sync(self._expire_old())

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    async def save_session_context(self, context: dict) -> None:
        """
        Save the current session's context to the session store.

        Args:
            context: A dict containing arbitrary session state (e.g., current task,
                     app state, conversation history, goals).
        """
        session_entry = {
            "session_id": self._session_id,
            "timestamp": _now_iso(),
            "context": context,
            "memory_count": len(self._memories),
        }

        with self._lock:
            # Check if this session already has an entry
            existing_idx = None
            for i, s in enumerate(self._sessions):
                if s.get("session_id") == self._session_id:
                    existing_idx = i
                    break

            if existing_idx is not None:
                self._sessions[existing_idx] = session_entry
            else:
                self._sessions.append(session_entry)

            # Trim
            if len(self._sessions) > MAX_SESSIONS:
                self._sessions = self._sessions[-MAX_SESSIONS:]

            self._force_save()

        logger.info("Saved session context for session %s.", self._session_id[:8])

    def save_session_context_sync(self, context: dict) -> None:
        """Synchronous wrapper for save_session_context()."""
        _run_sync(self.save_session_context(context))

    async def restore_session_context(self, session_id: str) -> dict:
        """
        Restore context from a previous session.

        Args:
            session_id: The session ID to restore.

        Returns:
            The context dict, or an empty dict if not found.
        """
        with self._lock:
            for entry in reversed(self._sessions):
                if entry.get("session_id") == session_id:
                    logger.info("Restored session context for session %s.", session_id[:8])
                    return entry.get("context", {})

        logger.warning("Session %s not found.", session_id[:8])
        return {}

    def restore_session_context_sync(self, session_id: str) -> dict:
        """Synchronous wrapper for restore_session_context()."""
        return _run_sync(self.restore_session_context(session_id))

    async def get_session_history(self, limit: int = 20) -> List[dict]:
        """
        List previous sessions, most recent first.

        Returns a list of session entry dicts with: session_id, timestamp,
        memory_count, and context keys.
        """
        with self._lock:
            sessions = list(reversed(self._sessions))[:limit]

        # Return summaries (omit full context for brevity)
        summaries: List[dict] = []
        for entry in sessions:
            summary = {
                "session_id": entry.get("session_id", ""),
                "timestamp": entry.get("timestamp", ""),
                "memory_count": entry.get("memory_count", 0),
                "context_keys": list(entry.get("context", {}).keys()),
            }
            summaries.append(summary)

        return summaries

    def get_session_history_sync(self, limit: int = 20) -> List[dict]:
        """Synchronous wrapper for get_session_history()."""
        return _run_sync(self.get_session_history(limit))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> MemoryStats:
        """
        Compute and return current memory statistics.
        Also persists the stats snapshot to disk.
        """
        with self._lock:
            total = len(self._memories)

            # By type
            by_type: Dict[str, int] = {}
            for mem in self._memories.values():
                type_key = mem.type.value if isinstance(mem.type, MemoryType) else str(mem.type)
                by_type[type_key] = by_type.get(type_key, 0) + 1

            # By priority
            by_priority: Dict[str, int] = {}
            for mem in self._memories.values():
                p_key = mem.priority.value if isinstance(mem.priority, MemoryPriority) else str(mem.priority)
                by_priority[p_key] = by_priority.get(p_key, 0) + 1

            # Total accesses
            total_accesses = sum(m.access_count for m in self._memories.values())

            # Oldest and newest
            sorted_by_time = sorted(
                self._memories.values(),
                key=lambda m: m.created_at or "",
            )
            oldest = sorted_by_time[0].created_at if sorted_by_time else ""
            newest = sorted_by_time[-1].created_at if sorted_by_time else ""

            # Average relevance
            if total > 0:
                avg_rel = sum(m.relevance_score for m in self._memories.values()) / total
            else:
                avg_rel = 0.0

            # Expired count (still in store)
            expired = sum(1 for m in self._memories.values() if m.is_expired())

            # Consolidation count
            consolidations = len(self._consolidation_log)

        stats = MemoryStats(
            total_memories=total,
            by_type=by_type,
            by_priority=by_priority,
            total_accesses=total_accesses,
            oldest_memory=oldest,
            newest_memory=newest,
            avg_relevance=round(avg_rel, 4),
            consolidations_run=consolidations,
            expired_count=expired,
        )

        # Persist snapshot
        _save_json(self._stats_path, stats.to_dict())

        return stats

    def get_stats_sync(self) -> MemoryStats:
        """Synchronous wrapper for get_stats()."""
        return _run_sync(self.get_stats())

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    async def export_memories(
        self,
        path: Union[str, Path],
        types: Optional[List[MemoryType]] = None,
        since: Optional[str] = None,
    ) -> int:
        """
        Export memories to a JSON file.

        Args:
            path:   Output file path.
            types:  Only export these memory types (None = all).
            since:  Only export memories created after this ISO timestamp.

        Returns:
            Number of memories exported.
        """
        export_path = Path(path)

        with self._lock:
            candidates = list(self._memories.values())

            # Filter by type
            if types:
                type_values = {t.value for t in types}
                candidates = [
                    m for m in candidates
                    if (m.type.value if isinstance(m.type, MemoryType) else str(m.type)) in type_values
                ]

            # Filter by time
            if since:
                since_dt = _parse_iso(since)
                if since_dt:
                    candidates = [
                        m for m in candidates
                        if _parse_iso(m.created_at) is not None
                        and _parse_iso(m.created_at) >= since_dt
                    ]

            # Limit export size
            candidates = candidates[:MAX_EXPORT_BATCH]

            export_data = {
                "exported_at": _now_iso(),
                "session_id": self._session_id,
                "count": len(candidates),
                "memories": [m.to_dict() for m in candidates],
            }

        _save_json(export_path, export_data)
        logger.info("Exported %d memories to %s.", len(candidates), export_path)
        return len(candidates)

    def export_memories_sync(
        self,
        path: Union[str, Path],
        types: Optional[List[MemoryType]] = None,
        since: Optional[str] = None,
    ) -> int:
        """Synchronous wrapper for export_memories()."""
        return _run_sync(self.export_memories(path, types, since))

    async def import_memories(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
    ) -> int:
        """
        Import memories from a JSON file.

        Args:
            path:      Input file path.
            overwrite: If True, overwrite existing memories with same ID.
                       If False, skip duplicates.

        Returns:
            Number of memories imported.
        """
        import_path = Path(path)

        raw = _load_json(import_path, default={})
        if not isinstance(raw, dict):
            logger.error("Invalid import file format: expected dict.")
            return 0

        memories_data = raw.get("memories", [])
        if not isinstance(memories_data, list):
            logger.error("Invalid import file: 'memories' field must be a list.")
            return 0

        imported = 0
        with self._lock:
            for mdata in memories_data:
                try:
                    mem = Memory.from_dict(mdata)
                except Exception as exc:
                    logger.warning("Skipping invalid memory during import: %s", exc)
                    continue

                if mem.id in self._memories and not overwrite:
                    logger.debug("Skipping duplicate memory %s.", mem.id[:8])
                    continue

                # If overwriting, remove old index entries first
                if mem.id in self._memories:
                    self._remove_from_index(mem.id)

                self._memories[mem.id] = mem
                self._add_to_index(mem)
                imported += 1

            if imported > 0:
                self._force_save()

        logger.info("Imported %d memories from %s.", imported, import_path)
        return imported

    def import_memories_sync(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
    ) -> int:
        """Synchronous wrapper for import_memories()."""
        return _run_sync(self.import_memories(path, overwrite))

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    async def count_by_type(self) -> Dict[str, int]:
        """Return a dict of memory type -> count."""
        with self._lock:
            counts: Dict[str, int] = {}
            for mem in self._memories.values():
                type_key = mem.type.value if isinstance(mem.type, MemoryType) else str(mem.type)
                counts[type_key] = counts.get(type_key, 0) + 1
        return counts

    async def count_by_tag(self, limit: int = 50) -> List[Tuple[str, int]]:
        """Return the most common tags and their counts."""
        with self._lock:
            tag_counter: Counter = Counter()
            for mem in self._memories.values():
                for tag in mem.tags:
                    tag_counter[tag.lower().strip()] += 1
        return tag_counter.most_common(limit)

    async def find_linked(self, memory_id: str, depth: int = 1) -> List[Memory]:
        """
        Find memories linked to the given memory, up to the specified depth.

        Args:
            memory_id: Starting memory ID.
            depth:     How many hops to follow (1 = direct links only).

        Returns:
            List of linked Memory objects (not including the source).
        """
        visited: Set[str] = {memory_id}
        current_layer: Set[str] = {memory_id}
        results: List[Memory] = []

        with self._lock:
            for _ in range(depth):
                next_layer: Set[str] = set()
                for mid in current_layer:
                    mem = self._memories.get(mid)
                    if mem is None:
                        continue
                    for linked_id in mem.linked_memories:
                        if linked_id not in visited:
                            visited.add(linked_id)
                            next_layer.add(linked_id)
                            linked_mem = self._memories.get(linked_id)
                            if linked_mem is not None:
                                results.append(linked_mem)
                current_layer = next_layer

        return results

    def find_linked_sync(self, memory_id: str, depth: int = 1) -> List[Memory]:
        """Synchronous wrapper for find_linked()."""
        return _run_sync(self.find_linked(memory_id, depth))

    async def search_metadata(self, key: str, value: Any, limit: int = 20) -> List[Memory]:
        """
        Search memories by metadata key-value pair.

        Args:
            key:   Metadata key to search.
            value: Value to match (uses equality comparison).
            limit: Max results.

        Returns:
            Matching memories sorted by relevance.
        """
        with self._lock:
            matches: List[Memory] = []
            for mem in self._memories.values():
                if mem.is_expired() or mem.consolidated:
                    continue
                if key in mem.metadata and mem.metadata[key] == value:
                    matches.append(mem)

            matches.sort(key=lambda m: -m.relevance_score)
            return matches[:limit]

    def search_metadata_sync(self, key: str, value: Any, limit: int = 20) -> List[Memory]:
        """Synchronous wrapper for search_metadata()."""
        return _run_sync(self.search_metadata(key, value, limit))

    async def clear_all(self) -> int:
        """
        Remove ALL memories from the store. Use with caution.
        Returns the number of memories removed.
        """
        with self._lock:
            count = len(self._memories)
            self._memories.clear()
            self._tag_index.clear()
            self._type_index.clear()
            self._force_save()

        logger.warning("Cleared all %d memories from store.", count)
        return count

    def clear_all_sync(self) -> int:
        """Synchronous wrapper for clear_all()."""
        return _run_sync(self.clear_all())

    async def prune(self) -> Dict[str, int]:
        """
        Run maintenance: expire old memories, decay relevance, rebuild indices.
        Returns a dict summarizing what was done.
        """
        expired = await self._expire_old()
        decayed = self._decay_relevance()

        with self._lock:
            self._build_indices()
            self._force_save()

        result = {
            "expired_removed": expired,
            "relevance_decayed": decayed,
            "total_remaining": len(self._memories),
        }

        logger.info(
            "Prune complete: expired=%d, decayed=%d, remaining=%d",
            expired,
            decayed,
            len(self._memories),
        )
        return result

    def prune_sync(self) -> Dict[str, int]:
        """Synchronous wrapper for prune()."""
        return _run_sync(self.prune())

    async def get_all_tags(self) -> List[str]:
        """Return a sorted list of all unique tags in the store."""
        with self._lock:
            return sorted(self._tag_index.keys())

    def get_all_tags_sync(self) -> List[str]:
        """Synchronous wrapper for get_all_tags()."""
        return _run_sync(self.get_all_tags())

    async def get_memories_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[Memory]:
        """Retrieve all memories created in a specific session."""
        with self._lock:
            candidates = [
                m for m in self._memories.values()
                if m.session_id == session_id
            ]
            candidates.sort(key=lambda m: m.created_at or "")
            return candidates[:limit]

    def get_memories_by_session_sync(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[Memory]:
        """Synchronous wrapper for get_memories_by_session()."""
        return _run_sync(self.get_memories_by_session(session_id, limit))


# ===================================================================
# SINGLETON
# ===================================================================

_instance: Optional[AgentMemory] = None


def get_memory(config: Optional[dict] = None) -> AgentMemory:
    """
    Get the global AgentMemory singleton.

    Creates the instance on first call, loading persisted state from disk.

    Args:
        config: Optional dict with keys:
                - data_dir: Path or str for data directory
                - session_id: str for session identification
    """
    global _instance
    if _instance is None:
        if config:
            _instance = AgentMemory(
                data_dir=config.get("data_dir"),
                session_id=config.get("session_id"),
            )
        else:
            _instance = AgentMemory()
    return _instance


# ===================================================================
# CLI FORMATTING
# ===================================================================

def _format_table(headers: List[str], rows: List[List[str]], max_col_width: int = 40) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    # Truncate long values
    truncated_rows: List[List[str]] = []
    for row in rows:
        truncated_rows.append([
            val[:max_col_width - 3] + "..." if len(val) > max_col_width else val
            for val in row
        ])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in truncated_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    # Build format string
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in truncated_rows:
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))

    return "\n".join(lines)


def _format_memory_detail(mem: Memory) -> str:
    """Format a single memory for detailed display."""
    type_val = mem.type.value if isinstance(mem.type, MemoryType) else str(mem.type)
    priority_val = mem.priority.value if isinstance(mem.priority, MemoryPriority) else str(mem.priority)

    lines = [
        f"  ID:           {mem.id}",
        f"  Type:         {type_val}",
        f"  Priority:     {priority_val}",
        f"  Relevance:    {mem.relevance_score:.4f}",
        f"  Access Count: {mem.access_count}",
        f"  Tags:         {', '.join(mem.tags) if mem.tags else 'none'}",
        f"  Created:      {mem.created_at}",
        f"  Last Access:  {mem.last_accessed or 'never'}",
        f"  Expires:      {mem.expires_at or 'never'}",
        f"  Source:       {mem.source_module or 'unknown'}",
        f"  Session:      {mem.session_id[:8] if mem.session_id else 'unknown'}",
        f"  Consolidated: {mem.consolidated}",
        f"  Linked:       {len(mem.linked_memories)} memories",
        f"  Content:",
    ]

    # Wrap content
    content = mem.content
    if len(content) > 500:
        content = content[:497] + "..."
    for line in content.split("\n"):
        lines.append(f"    {line}")

    if mem.summary:
        lines.append(f"  Summary:")
        summary = mem.summary
        if len(summary) > 300:
            summary = summary[:297] + "..."
        for line in summary.split("\n"):
            lines.append(f"    {line}")

    return "\n".join(lines)


# ===================================================================
# CLI COMMANDS
# ===================================================================

def _cmd_store(args: argparse.Namespace) -> None:
    """Store a new memory from the command line."""
    memory = get_memory()

    # Parse type
    try:
        mem_type = MemoryType(args.type)
    except ValueError:
        print(f"Invalid type: {args.type}")
        print(f"Valid types: {', '.join(t.value for t in MemoryType)}")
        return

    # Parse priority
    try:
        mem_priority = MemoryPriority(args.priority)
    except ValueError:
        print(f"Invalid priority: {args.priority}")
        print(f"Valid priorities: {', '.join(p.value for p in MemoryPriority)}")
        return

    # Parse tags
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    # Parse TTL
    ttl = int(args.ttl) if args.ttl else None

    # Parse metadata
    metadata: Dict[str, Any] = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print(f"Invalid metadata JSON: {args.metadata}")
            return

    mem = memory.store_sync(
        content=args.content,
        type=mem_type,
        tags=tags,
        metadata=metadata,
        priority=mem_priority,
        source_module=args.source or "cli",
        ttl_hours=ttl,
    )

    print(f"\n  Stored memory: {mem.id}")
    print(f"  Type: {mem_type.value} | Priority: {mem_priority.value}")
    print(f"  Tags: {', '.join(tags) if tags else 'none'}")
    print(f"  Expires: {mem.expires_at or 'never'}")
    print()


def _cmd_recall(args: argparse.Namespace) -> None:
    """Search memories from the command line."""
    memory = get_memory()

    # Parse types
    types: List[MemoryType] = []
    if args.type:
        try:
            types = [MemoryType(args.type)]
        except ValueError:
            print(f"Invalid type: {args.type}")
            return

    # Parse tags
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    # Parse min_priority
    min_priority: Optional[MemoryPriority] = None
    if args.min_priority:
        try:
            min_priority = MemoryPriority(args.min_priority)
        except ValueError:
            print(f"Invalid priority: {args.min_priority}")
            return

    query = MemoryQuery(
        text=args.query or "",
        types=types,
        tags=tags,
        limit=args.limit,
        min_priority=min_priority,
        include_expired=args.include_expired,
        include_consolidated=args.include_consolidated,
    )

    results = memory.recall_sync(query)

    if not results:
        print("No memories found matching query.")
        return

    if args.detail and len(results) == 1:
        print(f"\n  Memory Detail\n  {'=' * 50}")
        print(_format_memory_detail(results[0]))
        print()
        return

    headers = ["ID", "Type", "Priority", "Relevance", "Tags", "Created", "Content"]
    rows: List[List[str]] = []
    for mem in results:
        type_val = mem.type.value if isinstance(mem.type, MemoryType) else str(mem.type)
        priority_val = mem.priority.value if isinstance(mem.priority, MemoryPriority) else str(mem.priority)
        content_preview = mem.content[:60].replace("\n", " ")
        if len(mem.content) > 60:
            content_preview += "..."

        created_display = ""
        if mem.created_at:
            created_dt = _parse_iso(mem.created_at)
            if created_dt:
                created_display = created_dt.strftime("%m/%d %H:%M")

        rows.append([
            mem.id[:8] + "...",
            type_val,
            priority_val,
            f"{mem.relevance_score:.3f}",
            ",".join(mem.tags[:3]),
            created_display,
            content_preview,
        ])

    print(f"\n  Memory Search Results  --  {len(results)} memories\n")
    print(_format_table(headers, rows, max_col_width=50))
    print()


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show memory statistics."""
    memory = get_memory()
    stats = memory.get_stats_sync()

    print(f"\n  Agent Memory Statistics")
    print(f"  {'=' * 40}")
    print(f"  Total Memories:     {stats.total_memories}")
    print(f"  Total Accesses:     {stats.total_accesses}")
    print(f"  Avg Relevance:      {stats.avg_relevance:.4f}")
    print(f"  Consolidations:     {stats.consolidations_run}")
    print(f"  Expired (pending):  {stats.expired_count}")
    print(f"  Oldest Memory:      {stats.oldest_memory or 'none'}")
    print(f"  Newest Memory:      {stats.newest_memory or 'none'}")

    if stats.by_type:
        print(f"\n  By Type:")
        for type_key, count in sorted(stats.by_type.items(), key=lambda x: -x[1]):
            print(f"    {type_key:<20} {count}")

    if stats.by_priority:
        print(f"\n  By Priority:")
        for p_key, count in sorted(stats.by_priority.items()):
            print(f"    {p_key:<20} {count}")

    print()


def _cmd_consolidate(args: argparse.Namespace) -> None:
    """Run memory consolidation."""
    memory = get_memory()

    # Parse strategy
    try:
        strategy = ConsolidationStrategy(args.strategy)
    except ValueError:
        print(f"Invalid strategy: {args.strategy}")
        print(f"Valid: {', '.join(s.value for s in ConsolidationStrategy)}")
        return

    # Parse type filter
    type_filter: Optional[MemoryType] = None
    if args.type:
        try:
            type_filter = MemoryType(args.type)
        except ValueError:
            print(f"Invalid type: {args.type}")
            return

    result = memory.consolidate_sync(
        strategy=strategy,
        type_filter=type_filter,
        age_days=args.age_days,
        max_batch=args.max_batch,
    )

    print(f"\n  Consolidation Complete")
    print(f"  {'=' * 40}")
    print(f"  Strategy:     {result.strategy.value}")
    print(f"  Input Count:  {result.input_count}")
    print(f"  Output Count: {result.output_count}")
    print(f"  Affected:     {len(result.memories_affected)} memories")
    print(f"  Summary:      {result.summary}")
    print()


def _cmd_export(args: argparse.Namespace) -> None:
    """Export memories to a file."""
    memory = get_memory()

    # Parse types
    types: Optional[List[MemoryType]] = None
    if args.type:
        try:
            types = [MemoryType(args.type)]
        except ValueError:
            print(f"Invalid type: {args.type}")
            return

    count = memory.export_memories_sync(
        path=args.path,
        types=types,
        since=args.since,
    )

    print(f"Exported {count} memories to {args.path}")


def _cmd_import(args: argparse.Namespace) -> None:
    """Import memories from a file."""
    memory = get_memory()

    count = memory.import_memories_sync(
        path=args.path,
        overwrite=args.overwrite,
    )

    print(f"Imported {count} memories from {args.path}")


def _cmd_session(args: argparse.Namespace) -> None:
    """Show or restore session context."""
    memory = get_memory()

    if args.restore:
        context = memory.restore_session_context_sync(args.restore)
        if context:
            print(f"\n  Restored Session: {args.restore[:8]}...")
            print(f"  Context Keys: {', '.join(context.keys())}")
            print(f"\n  Full Context:")
            print(json.dumps(context, indent=2, default=str))
        else:
            print(f"Session {args.restore[:8]}... not found.")
        return

    # Show session history
    sessions = memory.get_session_history_sync(limit=args.limit)
    if not sessions:
        print("No session history found.")
        return

    headers = ["Session ID", "Timestamp", "Memories", "Context Keys"]
    rows: List[List[str]] = []
    for entry in sessions:
        rows.append([
            entry["session_id"][:12] + "...",
            entry.get("timestamp", "")[:19],
            str(entry.get("memory_count", 0)),
            ", ".join(entry.get("context_keys", [])),
        ])

    print(f"\n  Session History  --  {len(sessions)} sessions\n")
    print(_format_table(headers, rows, max_col_width=50))
    print(f"\n  Current Session: {memory.session_id}")
    print()


def _cmd_prune(args: argparse.Namespace) -> None:
    """Remove expired memories and run maintenance."""
    memory = get_memory()

    result = memory.prune_sync()

    print(f"\n  Prune Complete")
    print(f"  {'=' * 40}")
    print(f"  Expired Removed:     {result['expired_removed']}")
    print(f"  Relevance Decayed:   {result['relevance_decayed']}")
    print(f"  Total Remaining:     {result['total_remaining']}")
    print()


def _cmd_tags(args: argparse.Namespace) -> None:
    """List all tags and their counts."""
    memory = get_memory()

    tag_counts = _run_sync(memory.count_by_tag(limit=args.limit))

    if not tag_counts:
        print("No tags found.")
        return

    headers = ["Tag", "Count"]
    rows = [[tag, str(count)] for tag, count in tag_counts]

    print(f"\n  Memory Tags  --  {len(tag_counts)} tags\n")
    print(_format_table(headers, rows))
    print()


def _cmd_detail(args: argparse.Namespace) -> None:
    """Show detailed view of a single memory."""
    memory = get_memory()

    # Try to find by ID (full or prefix)
    mem: Optional[Memory] = None
    full_id = args.memory_id

    # Try exact match first
    mem = memory.get_sync(full_id)

    # Try prefix match
    if mem is None:
        with memory._lock:
            for mid, m in memory._memories.items():
                if mid.startswith(full_id):
                    mem = m
                    break

    if mem is None:
        print(f"Memory not found: {full_id}")
        return

    print(f"\n  Memory Detail\n  {'=' * 50}")
    print(_format_memory_detail(mem))

    # Show linked memories
    linked = memory.find_linked_sync(mem.id, depth=1)
    if linked:
        print(f"\n  Linked Memories ({len(linked)}):")
        for lm in linked:
            lm_type = lm.type.value if isinstance(lm.type, MemoryType) else str(lm.type)
            content_preview = lm.content[:60].replace("\n", " ")
            print(f"    {lm.id[:8]}... [{lm_type}] {content_preview}")

    print()


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the agent memory module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="agent_memory",
        description="OpenClaw Empire Agent Memory — Long-Term Memory Store",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # store
    sp_store = subparsers.add_parser("store", help="Store a new memory")
    sp_store.add_argument("content", type=str, help="Memory content text")
    sp_store.add_argument(
        "--type", type=str, default="fact",
        help=f"Memory type ({', '.join(t.value for t in MemoryType)})",
    )
    sp_store.add_argument(
        "--priority", type=str, default="normal",
        help=f"Priority ({', '.join(p.value for p in MemoryPriority)})",
    )
    sp_store.add_argument("--tags", type=str, default="", help="Comma-separated tags")
    sp_store.add_argument("--source", type=str, default="", help="Source module name")
    sp_store.add_argument("--ttl", type=str, default=None, help="TTL in hours (overrides priority default)")
    sp_store.add_argument("--metadata", type=str, default=None, help="JSON metadata string")
    sp_store.set_defaults(func=_cmd_store)

    # recall
    sp_recall = subparsers.add_parser("recall", help="Search memories")
    sp_recall.add_argument("query", type=str, nargs="?", default="", help="Search text")
    sp_recall.add_argument("--type", type=str, default=None, help="Filter by memory type")
    sp_recall.add_argument("--tags", type=str, default=None, help="Comma-separated tag filter")
    sp_recall.add_argument("--min-priority", type=str, default=None, help="Minimum priority level")
    sp_recall.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sp_recall.add_argument("--detail", action="store_true", help="Show detailed view (single result)")
    sp_recall.add_argument("--include-expired", action="store_true", help="Include expired memories")
    sp_recall.add_argument("--include-consolidated", action="store_true", help="Include consolidated memories")
    sp_recall.set_defaults(func=_cmd_recall)

    # stats
    sp_stats = subparsers.add_parser("stats", help="Show memory statistics")
    sp_stats.set_defaults(func=_cmd_stats)

    # consolidate
    sp_consolidate = subparsers.add_parser("consolidate", help="Run memory consolidation")
    sp_consolidate.add_argument(
        "--strategy", type=str, default="summarize",
        help=f"Strategy ({', '.join(s.value for s in ConsolidationStrategy)})",
    )
    sp_consolidate.add_argument("--type", type=str, default=None, help="Filter by memory type")
    sp_consolidate.add_argument("--age-days", type=int, default=14, help="Min age in days (default: 14)")
    sp_consolidate.add_argument("--max-batch", type=int, default=50, help="Max batch size (default: 50)")
    sp_consolidate.set_defaults(func=_cmd_consolidate)

    # export
    sp_export = subparsers.add_parser("export", help="Export memories to file")
    sp_export.add_argument("path", type=str, help="Output file path")
    sp_export.add_argument("--type", type=str, default=None, help="Filter by memory type")
    sp_export.add_argument("--since", type=str, default=None, help="Only export after this ISO timestamp")
    sp_export.set_defaults(func=_cmd_export)

    # import
    sp_import = subparsers.add_parser("import", help="Import memories from file")
    sp_import.add_argument("path", type=str, help="Input file path")
    sp_import.add_argument("--overwrite", action="store_true", help="Overwrite existing memories with same ID")
    sp_import.set_defaults(func=_cmd_import)

    # session
    sp_session = subparsers.add_parser("session", help="Show/restore session context")
    sp_session.add_argument("--restore", type=str, default=None, help="Session ID to restore")
    sp_session.add_argument("--limit", type=int, default=20, help="Max sessions to show (default: 20)")
    sp_session.set_defaults(func=_cmd_session)

    # prune
    sp_prune = subparsers.add_parser("prune", help="Remove expired memories and run maintenance")
    sp_prune.set_defaults(func=_cmd_prune)

    # tags
    sp_tags = subparsers.add_parser("tags", help="List all tags and counts")
    sp_tags.add_argument("--limit", type=int, default=50, help="Max tags to show (default: 50)")
    sp_tags.set_defaults(func=_cmd_tags)

    # detail
    sp_detail = subparsers.add_parser("detail", help="Show detailed view of a memory")
    sp_detail.add_argument("memory_id", type=str, help="Memory ID (full or prefix)")
    sp_detail.set_defaults(func=_cmd_detail)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


# ===================================================================
# MODULE ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
