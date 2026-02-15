"""
RAG Memory — TF-IDF Embedding-Based Semantic Search for OpenClaw Empire
========================================================================

Zero-cost, pure-Python TF-IDF engine with cosine similarity for semantic
search over agent_memory.  No external packages required — all computation
uses math.log, math.sqrt, and Python standard library only.

Builds a searchable TF-IDF index of every memory stored by the agent_memory
module and exposes search, context-building, and analytics APIs that can be
used directly by the autonomous agent or fed into LLM prompts as RAG context.

Capabilities:
    - Pure Python TF-IDF with cosine similarity (zero API cost)
    - Three search strategies: semantic, keyword, hybrid
    - Context builder that formats search results for LLM prompts
    - Conversation context from multi-turn message history
    - Similar-memory discovery for link suggestions
    - Vocabulary analytics (top IDF terms, coverage stats)
    - Persistent index with atomic JSON save/load
    - Full CLI with subcommands: index, search, context, similar, stats, terms

Data persisted to: data/rag/
    tfidf_index.json     — vocabulary, IDF weights, document vectors
    search_log.json      — search history and performance metrics
    index_meta.json      — indexing metadata (last rebuild, doc count)

Usage:
    from src.rag_memory import get_rag_memory

    rag = get_rag_memory()

    # Build the index from agent_memory
    rag.index_all_sync()

    # Semantic search
    results = rag.semantic_search_sync("instagram login 2fa", limit=5)

    # Build LLM context
    ctx = rag.build_context_sync("how to log into instagram", max_tokens=2000)
    print(ctx.context_string)   # formatted for prompt injection

    # Find similar memories
    similar = rag.get_similar_memories("some-memory-id", limit=5)

CLI:
    python -m src.rag_memory index
    python -m src.rag_memory search --query "instagram login" --strategy hybrid --limit 10
    python -m src.rag_memory context --query "how to post on tiktok" --max-tokens 2000
    python -m src.rag_memory similar --memory-id UUID --limit 5
    python -m src.rag_memory stats
    python -m src.rag_memory terms --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
import sys
import time
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("rag_memory")

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

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")

RAG_DATA_DIR = BASE_DIR / "data" / "rag"
TFIDF_INDEX_FILE = RAG_DATA_DIR / "tfidf_index.json"
SEARCH_LOG_FILE = RAG_DATA_DIR / "search_log.json"
INDEX_META_FILE = RAG_DATA_DIR / "index_meta.json"

# Ensure data directory exists on import
RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Index limits
MAX_VOCABULARY_SIZE = 100_000
MAX_SEARCH_LOG_ENTRIES = 5000
MAX_DOCUMENT_TOKENS = 10_000       # per document tokenization cap

# Search defaults
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_MIN_SCORE = 0.05
DEFAULT_CONTEXT_MAX_TOKENS = 2000
DEFAULT_SEMANTIC_WEIGHT = 0.7      # for hybrid search

# Token estimation
CHARS_PER_TOKEN = 4                # rough approximation

# Stemmer minimum word length
STEM_MIN_LENGTH = 3


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

class EmbeddingMethod(str, Enum):
    """Method used for computing document embeddings."""
    TFIDF = "tfidf"                # Zero-cost, stdlib only (default)
    ANTHROPIC = "anthropic"        # Higher quality, API cost


class SearchStrategy(str, Enum):
    """Strategy for searching the memory index."""
    SEMANTIC = "semantic"          # TF-IDF cosine similarity
    KEYWORD = "keyword"            # Exact/fuzzy keyword match
    HYBRID = "hybrid"              # Combine semantic + keyword


# ===================================================================
# DATA CLASSES
# ===================================================================

@dataclass
class SearchResult:
    """A single search result from the RAG memory index."""
    memory_id: str = ""
    content: str = ""
    score: float = 0.0             # 0-1 similarity
    memory_type: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SearchResult:
        """Deserialize from a plain dict loaded from JSON."""
        data = dict(data)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class RAGContext:
    """Context object returned by the RAG context builder."""
    query: str = ""
    results: List[SearchResult] = field(default_factory=list)
    context_string: str = ""       # formatted for LLM prompt
    token_estimate: int = 0
    strategy_used: SearchStrategy = SearchStrategy.HYBRID
    search_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        d = asdict(self)
        d["strategy_used"] = self.strategy_used.value
        d["results"] = [r if isinstance(r, dict) else asdict(r) for r in self.results]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RAGContext:
        """Deserialize from a plain dict."""
        data = dict(data)
        if "strategy_used" in data:
            try:
                data["strategy_used"] = SearchStrategy(data["strategy_used"])
            except ValueError:
                data["strategy_used"] = SearchStrategy.HYBRID
        if "results" in data and isinstance(data["results"], list):
            data["results"] = [
                SearchResult.from_dict(r) if isinstance(r, dict) else r
                for r in data["results"]
            ]
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class SearchLogEntry:
    """Record of a search operation for analytics."""
    query: str = ""
    strategy: str = ""
    result_count: int = 0
    top_score: float = 0.0
    search_time_ms: float = 0.0
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# STOPWORDS — comprehensive English stopword list
# ===================================================================

STOPWORDS: Set[str] = {
    # Articles and determiners
    "a", "an", "the", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "every", "each", "all", "both",
    "few", "more", "most", "other", "such", "what", "which",
    "whose", "whatever", "whichever",

    # Pronouns
    "i", "me", "we", "us", "you", "he", "him", "she",
    "it", "they", "them", "myself", "yourself", "himself",
    "herself", "itself", "ourselves", "yourselves", "themselves",

    # Prepositions
    "about", "above", "across", "after", "against", "along",
    "among", "around", "at", "before", "behind", "below",
    "beneath", "beside", "between", "beyond", "by", "down",
    "during", "except", "for", "from", "in", "inside", "into",
    "like", "near", "of", "off", "on", "onto", "out",
    "outside", "over", "past", "since", "through", "throughout",
    "to", "toward", "towards", "under", "underneath", "until",
    "unto", "up", "upon", "with", "within", "without",

    # Conjunctions
    "and", "but", "or", "nor", "for", "yet", "so",
    "although", "because", "if", "unless", "while", "whereas",
    "whether", "though", "even", "once", "than",

    # Common verbs (be, have, do, auxiliaries)
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "done", "will", "would", "shall", "should", "may", "might",
    "can", "could", "must", "need", "dare", "ought",

    # Common adverbs and particles
    "not", "very", "really", "just", "also", "too", "quite",
    "rather", "almost", "already", "always", "never", "often",
    "sometimes", "usually", "here", "there", "where", "when",
    "how", "why", "then", "now", "still", "only", "well",
    "much", "many", "enough", "else", "ever", "perhaps",
    "maybe", "again", "further", "soon", "away", "back",
    "ago", "hence", "therefore", "thus", "however", "moreover",
    "furthermore", "nevertheless", "nonetheless", "instead",
    "meanwhile", "otherwise", "anyway", "besides",

    # Common words with low information content
    "get", "got", "make", "made", "go", "went", "gone",
    "come", "came", "take", "took", "taken", "give", "gave",
    "given", "see", "saw", "seen", "know", "knew", "known",
    "think", "thought", "say", "said", "tell", "told",
    "find", "found", "put", "set", "keep", "kept", "let",
    "seem", "help", "show", "turn", "move", "play", "run",
    "call", "try", "ask", "work", "use", "used",

    # Other function words
    "own", "same", "able", "thing", "things", "way", "ways",
    "part", "good", "new", "old", "big", "small", "long",
    "great", "little", "right", "left", "first", "last",
    "next", "sure", "true", "real", "lot", "much",
    "one", "two", "three", "four", "five",
}


# ===================================================================
# SIMPLE SUFFIX-STRIPPING STEMMER (Porter-lite)
# ===================================================================

class SimpleStemmer:
    """
    Lightweight suffix-stripping stemmer for English text.

    Not a full Porter stemmer — just strips common suffixes to normalize
    words for TF-IDF matching.  Designed for zero external dependencies.

    Rules (applied in order, first match wins):
        1.  -ational  -> -ate      (relational -> relate)
        2.  -tional   -> -tion     (conditional -> condition)
        3.  -iveness  -> -ive      (effectiveness -> effective)
        4.  -fulness  -> -ful      (hopefulness -> hopeful)
        5.  -ousness  -> -ous      (consciousness -> conscious)
        6.  -ization  -> -ize      (organization -> organize)
        7.  -ation    -> -ate      (information -> informate) [close enough]
        8.  -ement    -> -e        (replacement -> replace)
        9.  -ments    -> -ment     (arguments -> argument)
        10. -ness     -> (remove)  (darkness -> dark)
        11. -ment     -> (remove)  (government -> govern)
        12. -ings     -> (remove)  (settings -> sett) [acceptable for matching]
        13. -ling     -> -le       (handling -> handle)
        14. -ting     -> -t        (setting -> set)  [skip if too short]
        15. -ing      -> (remove)  (running -> runn)
        16. -edly     -> -e        (reportedly -> reporte)
        17. -edly     -> (remove)  (admittedly -> admitted) [fallback]
        18. -ably     -> -able     (presumably -> presumable)
        19. -ibly     -> -ible     (possibly -> possible)
        20. -ally     -> -al       (historically -> historical)
        21. -ily      -> -y        (happily -> happy)
        22. -ly       -> (remove)  (quickly -> quick)
        23. -ies      -> -y        (stories -> story)
        24. -ied      -> -y        (tried -> try)
        25. -ier      -> -y        (happier -> happy)
        26. -iest     -> -y        (happiest -> happy)
        27. -ive      -> (remove)  (active -> act)
        28. -ise/-ize -> (remove)  (organize -> organ)
        29. -ous      -> (remove)  (dangerous -> danger)
        30. -ful      -> (remove)  (hopeful -> hope)
        31. -less     -> (remove)  (careless -> care)
        32. -able     -> (remove)  (readable -> read)
        33. -ible     -> (remove)  (possible -> poss) [acceptable]
        34. -tion     -> -t        (action -> act)
        35. -sion     -> -s        (decision -> decis)
        36. -ence     -> (remove)  (difference -> differ)
        37. -ance     -> (remove)  (performance -> perform)
        38. -ers      -> -er       (workers -> worker)
        39. -er       -> (remove)  (worker -> work)
        40. -es       -> (remove)  (boxes -> box)
        41. -ed       -> (remove)  (worked -> work)
        42. -s        -> (remove)  (cats -> cat) [not after s]
    """

    # Suffix rules: (suffix, replacement, min_remaining_length)
    # min_remaining_length ensures we don't over-strip short words
    RULES: List[Tuple[str, str, int]] = [
        # Long compound suffixes first (order matters)
        ("ational", "ate", 2),
        ("tional", "tion", 2),
        ("iveness", "ive", 2),
        ("fulness", "ful", 2),
        ("ousness", "ous", 2),
        ("ization", "ize", 2),
        ("isation", "ise", 2),
        ("ation", "ate", 3),
        ("ement", "e", 3),
        ("ments", "ment", 3),
        ("ness", "", 3),
        ("ment", "", 3),
        ("ings", "", 3),
        ("ling", "le", 3),
        ("ting", "t", 3),
        ("ing", "", 3),
        ("edly", "e", 3),
        ("ably", "able", 2),
        ("ibly", "ible", 2),
        ("ally", "al", 3),
        ("ily", "y", 3),
        ("ly", "", 3),
        ("ies", "y", 2),
        ("ied", "y", 2),
        ("ier", "y", 2),
        ("iest", "y", 2),
        ("ive", "", 3),
        ("ize", "", 3),
        ("ise", "", 3),
        ("ous", "", 3),
        ("ful", "", 3),
        ("less", "", 3),
        ("able", "", 3),
        ("ible", "", 3),
        ("tion", "t", 2),
        ("sion", "s", 2),
        ("ence", "", 3),
        ("ance", "", 3),
        ("ers", "er", 2),
        ("er", "", 3),
        ("es", "", 3),
        ("ed", "", 3),
        ("s", "", 3),
    ]

    # Irregular stems that suffix stripping handles poorly
    IRREGULARS: Dict[str, str] = {
        "are": "be",
        "is": "be",
        "was": "be",
        "were": "be",
        "been": "be",
        "being": "be",
        "am": "be",
        "has": "have",
        "had": "have",
        "having": "have",
        "does": "do",
        "did": "do",
        "doing": "do",
        "done": "do",
        "goes": "go",
        "went": "go",
        "gone": "go",
        "going": "go",
        "came": "come",
        "coming": "come",
        "ran": "run",
        "running": "run",
        "saw": "see",
        "seen": "see",
        "seeing": "see",
        "knew": "know",
        "known": "know",
        "knowing": "know",
        "took": "take",
        "taken": "take",
        "taking": "take",
        "gave": "give",
        "given": "give",
        "giving": "give",
        "made": "make",
        "making": "make",
        "said": "say",
        "saying": "say",
        "told": "tell",
        "telling": "tell",
        "found": "find",
        "finding": "find",
        "thought": "think",
        "thinking": "think",
        "got": "get",
        "gotten": "get",
        "getting": "get",
        "kept": "keep",
        "keeping": "keep",
        "children": "child",
        "women": "woman",
        "men": "man",
        "people": "person",
        "mice": "mouse",
        "feet": "foot",
        "teeth": "tooth",
        "data": "datum",
        "analyses": "analysis",
        "indices": "index",
        "matrices": "matrix",
    }

    _cache: Dict[str, str] = {}

    @classmethod
    def stem(cls, word: str) -> str:
        """
        Stem a single word by stripping common suffixes.

        Args:
            word: Lowercase word to stem.

        Returns:
            Stemmed form of the word.
        """
        if not word or len(word) < STEM_MIN_LENGTH:
            return word

        # Check cache first
        if word in cls._cache:
            return cls._cache[word]

        original = word

        # Check irregulars
        if word in cls.IRREGULARS:
            result = cls.IRREGULARS[word]
            cls._cache[original] = result
            return result

        # Apply suffix rules — first match wins
        for suffix, replacement, min_remaining in cls.RULES:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if len(stem) >= min_remaining:
                    result = stem + replacement
                    # Avoid creating empty stems
                    if len(result) >= 2:
                        cls._cache[original] = result
                        return result
                # If remaining is too short, skip this rule and try next
                continue

        # No rule matched — return as-is
        cls._cache[original] = word
        return word

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the stemmer cache."""
        cls._cache.clear()

    @classmethod
    def cache_size(cls) -> int:
        """Return the number of cached stems."""
        return len(cls._cache)


# ===================================================================
# TF-IDF ENGINE — Pure Python
# ===================================================================

class TFIDFEngine:
    """
    Pure Python TF-IDF implementation with cosine similarity.

    Builds an inverted index of term frequencies and inverse document
    frequencies, then computes cosine similarity between query vectors
    and stored document vectors.

    All computation uses math.log and math.sqrt — no numpy, no sklearn,
    no external packages.

    Attributes:
        _vocabulary:        Dict mapping each term to a unique integer index.
        _idf:               Dict mapping each term to its IDF weight.
        _document_vectors:  Dict mapping doc_id to {term: tfidf_weight}.
        _document_norms:    Dict mapping doc_id to L2 norm of its TF-IDF vector.
        _document_tokens:   Dict mapping doc_id to list of tokens (for stats).
        _document_count:    Total number of indexed documents.
        _term_doc_counts:   Dict mapping each term to number of docs containing it.
        _raw_texts:         Dict mapping doc_id to original text (for keyword search).
    """

    def __init__(self) -> None:
        """Initialize an empty TF-IDF engine."""
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._document_vectors: Dict[str, Dict[str, float]] = {}
        self._document_norms: Dict[str, float] = {}
        self._document_tokens: Dict[str, List[str]] = {}
        self._document_count: int = 0
        self._term_doc_counts: Dict[str, int] = defaultdict(int)
        self._raw_texts: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._needs_idf_rebuild: bool = False
        self._stemmer = SimpleStemmer

        logger.debug("TFIDFEngine initialized (empty).")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize and normalize text for TF-IDF indexing.

        Steps:
            1. Lowercase the text.
            2. Replace non-alphanumeric characters with spaces.
            3. Split on whitespace.
            4. Remove tokens shorter than 2 characters.
            5. Remove stopwords.
            6. Apply simple suffix-stripping stemmer.
            7. Remove tokens that are purely numeric.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of normalized, stemmed tokens.
        """
        if not text:
            return []

        # Lowercase
        text = text.lower()

        # Replace non-alphanumeric with spaces (keep hyphens between words)
        text = re.sub(r"[^a-z0-9\-]", " ", text)

        # Split hyphenated words into components as well
        text = text.replace("-", " ")

        # Split on whitespace
        raw_tokens = text.split()

        # Filter, stem, and deduplicate-in-order
        tokens: List[str] = []
        for token in raw_tokens:
            # Skip short tokens
            if len(token) < 2:
                continue

            # Skip pure numbers
            if token.isdigit():
                continue

            # Skip stopwords
            if token in STOPWORDS:
                continue

            # Apply stemmer
            stemmed = self._stemmer.stem(token)

            # Skip if stemming produced something too short
            if len(stemmed) < 2:
                continue

            tokens.append(stemmed)

        # Cap token count to prevent memory issues on huge documents
        if len(tokens) > MAX_DOCUMENT_TOKENS:
            tokens = tokens[:MAX_DOCUMENT_TOKENS]

        return tokens

    # ------------------------------------------------------------------
    # Term Frequency
    # ------------------------------------------------------------------

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Compute term frequency for a list of tokens.

        Uses augmented (normalized) term frequency:
            tf(t, d) = 0.5 + 0.5 * (count(t, d) / max_count(d))

        This normalizes against document length bias while keeping
        the most frequent term's TF at 1.0.

        Args:
            tokens: List of stemmed tokens from _tokenize().

        Returns:
            Dict mapping each unique token to its TF weight.
        """
        if not tokens:
            return {}

        counts = Counter(tokens)
        max_count = max(counts.values())

        tf: Dict[str, float] = {}
        for term, count in counts.items():
            # Augmented TF avoids bias toward longer documents
            tf[term] = 0.5 + 0.5 * (count / max_count)

        return tf

    # ------------------------------------------------------------------
    # IDF computation
    # ------------------------------------------------------------------

    def _rebuild_idf(self) -> None:
        """
        Recompute inverse document frequency for all terms across the corpus.

        Uses smoothed IDF:
            idf(t) = log(1 + N / (1 + df(t)))

        where N is total documents and df(t) is number of documents
        containing term t.  The +1 in denominator prevents division by
        zero; the +1 inside log prevents negative IDF for very common terms.
        """
        with self._lock:
            n = max(self._document_count, 1)
            new_idf: Dict[str, float] = {}

            for term, doc_count in self._term_doc_counts.items():
                # Smoothed IDF
                new_idf[term] = math.log(1.0 + n / (1.0 + doc_count))

            self._idf = new_idf
            self._needs_idf_rebuild = False

            logger.debug(
                "IDF rebuilt: %d terms across %d documents.",
                len(self._idf),
                self._document_count,
            )

    def _rebuild_all_vectors(self) -> None:
        """
        Rebuild all document TF-IDF vectors and norms after IDF update.

        This is called after _rebuild_idf() to ensure all document vectors
        use the latest IDF weights.  Should be called during full reindex.
        """
        with self._lock:
            for doc_id, tokens in self._document_tokens.items():
                tf = self._compute_tf(tokens)
                vector: Dict[str, float] = {}

                for term, tf_weight in tf.items():
                    idf_weight = self._idf.get(term, 0.0)
                    tfidf = tf_weight * idf_weight
                    if tfidf > 0.0:
                        vector[term] = tfidf

                self._document_vectors[doc_id] = vector
                self._document_norms[doc_id] = self._compute_norm(vector)

    def _compute_norm(self, vector: Dict[str, float]) -> float:
        """
        Compute the L2 norm of a sparse vector.

        Args:
            vector: Dict mapping terms to weights.

        Returns:
            L2 norm (Euclidean length) of the vector.
        """
        if not vector:
            return 0.0
        return math.sqrt(sum(w * w for w in vector.values()))

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        """
        Add or update a document in the TF-IDF index.

        Tokenizes the text, computes TF, updates term-document counts,
        and stores the vector.  Marks IDF as needing rebuild.

        Args:
            doc_id: Unique identifier for the document.
            text:   Raw text content of the document.
        """
        # If document already exists, remove it first
        if doc_id in self._document_tokens:
            self.remove_document(doc_id)

        tokens = self._tokenize(text)
        if not tokens:
            # Store empty document so we know about it
            with self._lock:
                self._document_tokens[doc_id] = []
                self._document_vectors[doc_id] = {}
                self._document_norms[doc_id] = 0.0
                self._raw_texts[doc_id] = text
                self._document_count += 1
            return

        tf = self._compute_tf(tokens)

        # Update term-document counts
        unique_terms = set(tokens)

        with self._lock:
            for term in unique_terms:
                self._term_doc_counts[term] += 1
                # Add to vocabulary
                if term not in self._vocabulary:
                    if len(self._vocabulary) < MAX_VOCABULARY_SIZE:
                        self._vocabulary[term] = len(self._vocabulary)

            # Store tokens for later IDF rebuild
            self._document_tokens[doc_id] = tokens
            self._raw_texts[doc_id] = text
            self._document_count += 1
            self._needs_idf_rebuild = True

            # Compute initial vector with current IDF (may be stale)
            vector: Dict[str, float] = {}
            for term, tf_weight in tf.items():
                idf_weight = self._idf.get(term, 1.0)  # default 1.0 for new terms
                tfidf = tf_weight * idf_weight
                if tfidf > 0.0:
                    vector[term] = tfidf

            self._document_vectors[doc_id] = vector
            self._document_norms[doc_id] = self._compute_norm(vector)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the TF-IDF index.

        Updates term-document counts and removes the document's vector.

        Args:
            doc_id: Unique identifier for the document to remove.

        Returns:
            True if the document was found and removed, False otherwise.
        """
        with self._lock:
            if doc_id not in self._document_tokens:
                return False

            tokens = self._document_tokens[doc_id]
            unique_terms = set(tokens)

            # Decrement term-document counts
            for term in unique_terms:
                if term in self._term_doc_counts:
                    self._term_doc_counts[term] -= 1
                    if self._term_doc_counts[term] <= 0:
                        del self._term_doc_counts[term]
                        # Note: we don't remove from vocabulary to keep indices stable

            del self._document_tokens[doc_id]
            self._document_vectors.pop(doc_id, None)
            self._document_norms.pop(doc_id, None)
            self._raw_texts.pop(doc_id, None)
            self._document_count = max(self._document_count - 1, 0)
            self._needs_idf_rebuild = True

        return True

    def has_document(self, doc_id: str) -> bool:
        """Check if a document is in the index."""
        return doc_id in self._document_tokens

    def document_ids(self) -> List[str]:
        """Return all document IDs in the index."""
        return list(self._document_tokens.keys())

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    def _cosine_similarity(
        self,
        query_vector: Dict[str, float],
        query_norm: float,
        doc_id: str,
    ) -> float:
        """
        Compute cosine similarity between a query vector and a document vector.

        cosine_sim(q, d) = (q . d) / (||q|| * ||d||)

        Args:
            query_vector: Sparse vector {term: weight} for the query.
            query_norm:   L2 norm of the query vector.
            doc_id:       ID of the document to compare against.

        Returns:
            Cosine similarity score in [0, 1].
        """
        doc_vector = self._document_vectors.get(doc_id, {})
        doc_norm = self._document_norms.get(doc_id, 0.0)

        if query_norm == 0.0 or doc_norm == 0.0:
            return 0.0

        # Dot product — iterate over the smaller vector for efficiency
        if len(query_vector) <= len(doc_vector):
            dot = sum(
                weight * doc_vector.get(term, 0.0)
                for term, weight in query_vector.items()
            )
        else:
            dot = sum(
                weight * query_vector.get(term, 0.0)
                for term, weight in doc_vector.items()
            )

        similarity = dot / (query_norm * doc_norm)

        # Clamp to [0, 1] (floating point can cause tiny overshoots)
        return max(0.0, min(1.0, similarity))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> List[Tuple[str, float]]:
        """
        Search the index for documents similar to the query.

        Tokenizes the query, builds a TF-IDF query vector using the
        current IDF weights, then computes cosine similarity against
        all indexed documents.

        Args:
            query:     Natural language search query.
            limit:     Maximum number of results to return.
            min_score: Minimum cosine similarity score to include.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        # Ensure IDF is up to date before searching
        if self._needs_idf_rebuild:
            self._rebuild_idf()
            self._rebuild_all_vectors()

        # Tokenize the query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build query TF vector
        query_tf = self._compute_tf(query_tokens)

        # Build query TF-IDF vector using corpus IDF
        query_vector: Dict[str, float] = {}
        for term, tf_weight in query_tf.items():
            idf_weight = self._idf.get(term, 0.0)
            if idf_weight > 0.0:
                query_vector[term] = tf_weight * idf_weight

        if not query_vector:
            # No query terms found in vocabulary — try partial matches
            return []

        query_norm = self._compute_norm(query_vector)
        if query_norm == 0.0:
            return []

        # Score all documents
        scores: List[Tuple[str, float]] = []
        for doc_id in self._document_vectors:
            score = self._cosine_similarity(query_vector, query_norm, doc_id)
            if score >= min_score:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        return scores[:limit]

    def keyword_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[Tuple[str, float]]:
        """
        Perform keyword-based search on raw document text.

        Uses substring matching and term overlap scoring rather than
        TF-IDF cosine similarity.  Good for exact phrase matching.

        Scoring:
            - Each query token found in the document contributes
              1.0 / total_query_tokens to the score.
            - Exact substring match of the full query adds a 0.3 bonus.
            - Score is clamped to [0, 1].

        Args:
            query:  Search query string.
            limit:  Maximum number of results.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            # Fall back to raw word splitting
            query_tokens = [w for w in query_lower.split() if len(w) >= 2]
        if not query_tokens:
            return []

        num_query_tokens = len(query_tokens)
        scores: List[Tuple[str, float]] = []

        for doc_id, raw_text in self._raw_texts.items():
            text_lower = raw_text.lower()
            doc_tokens_set = set(self._document_tokens.get(doc_id, []))

            score = 0.0

            # Token overlap scoring
            matched_tokens = 0
            for qt in query_tokens:
                if qt in doc_tokens_set:
                    matched_tokens += 1

            if matched_tokens > 0:
                score = matched_tokens / num_query_tokens

            # Exact substring bonus
            if query_lower in text_lower:
                score += 0.3

            # Partial phrase bonus (bigrams)
            if num_query_tokens >= 2:
                query_words = query_lower.split()
                for i in range(len(query_words) - 1):
                    bigram = query_words[i] + " " + query_words[i + 1]
                    if bigram in text_lower:
                        score += 0.1

            # Clamp score
            score = min(1.0, max(0.0, score))

            if score > 0.0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the TF-IDF index.

        Returns:
            Dict with keys: document_count, vocabulary_size, total_tokens,
            avg_document_length, idf_terms, needs_rebuild, stemmer_cache_size.
        """
        total_tokens = sum(len(tokens) for tokens in self._document_tokens.values())
        avg_length = total_tokens / max(self._document_count, 1)

        return {
            "document_count": self._document_count,
            "vocabulary_size": len(self._vocabulary),
            "total_tokens": total_tokens,
            "avg_document_length": round(avg_length, 2),
            "idf_terms": len(self._idf),
            "term_doc_count_entries": len(self._term_doc_counts),
            "needs_rebuild": self._needs_idf_rebuild,
            "stemmer_cache_size": self._stemmer.cache_size(),
        }

    def get_top_terms(self, limit: int = 50) -> List[Tuple[str, float]]:
        """
        Return the top terms by IDF weight (most discriminating terms).

        High IDF = rare term = more discriminating.  These are the terms
        that best distinguish between documents.

        Args:
            limit: Maximum number of terms to return.

        Returns:
            List of (term, idf_weight) tuples sorted by IDF descending.
        """
        if self._needs_idf_rebuild:
            self._rebuild_idf()

        sorted_terms = sorted(
            self._idf.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_terms[:limit]

    def get_document_terms(self, doc_id: str) -> List[Tuple[str, float]]:
        """
        Return the TF-IDF weighted terms for a specific document.

        Args:
            doc_id: Document identifier.

        Returns:
            List of (term, tfidf_weight) sorted by weight descending.
        """
        vector = self._document_vectors.get(doc_id, {})
        sorted_terms = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the TF-IDF engine state to a dictionary.

        Returns:
            Dict containing all engine state for JSON persistence.
        """
        return {
            "vocabulary": self._vocabulary,
            "idf": self._idf,
            "document_vectors": self._document_vectors,
            "document_norms": self._document_norms,
            "document_tokens": self._document_tokens,
            "document_count": self._document_count,
            "term_doc_counts": dict(self._term_doc_counts),
            "raw_texts": self._raw_texts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TFIDFEngine:
        """
        Deserialize a TF-IDF engine from a dictionary.

        Args:
            data: Dict from to_dict() / JSON load.

        Returns:
            Restored TFIDFEngine instance.
        """
        engine = cls()

        engine._vocabulary = data.get("vocabulary", {})
        engine._idf = data.get("idf", {})
        engine._document_vectors = data.get("document_vectors", {})
        engine._document_norms = {
            k: float(v) for k, v in data.get("document_norms", {}).items()
        }
        engine._document_tokens = data.get("document_tokens", {})
        engine._document_count = data.get("document_count", 0)
        engine._term_doc_counts = defaultdict(
            int, data.get("term_doc_counts", {})
        )
        engine._raw_texts = data.get("raw_texts", {})
        engine._needs_idf_rebuild = False

        logger.debug(
            "TFIDFEngine restored from dict: %d docs, %d vocab.",
            engine._document_count,
            len(engine._vocabulary),
        )

        return engine

    def clear(self) -> None:
        """Reset the engine to empty state."""
        with self._lock:
            self._vocabulary.clear()
            self._idf.clear()
            self._document_vectors.clear()
            self._document_norms.clear()
            self._document_tokens.clear()
            self._document_count = 0
            self._term_doc_counts.clear()
            self._raw_texts.clear()
            self._needs_idf_rebuild = False
            self._stemmer.clear_cache()

        logger.debug("TFIDFEngine cleared.")


# ===================================================================
# RAG MEMORY — Main Class
# ===================================================================

class RAGMemory:
    """
    Semantic search over agent memories for RAG context building.

    Builds and maintains a TF-IDF index of all memories from the
    agent_memory module.  Provides three search strategies (semantic,
    keyword, hybrid) and a context builder that formats results for
    injection into LLM prompts.

    Usage:
        rag = get_rag_memory()
        rag.index_all_sync()

        # Search
        results = rag.semantic_search_sync("instagram login")

        # Build context for LLM
        ctx = rag.build_context_sync("how to post on tiktok")
        system_prompt = f"Use this context:\\n{ctx.context_string}"

    Attributes:
        _tfidf:          TFIDFEngine instance.
        _indexed_count:  Number of memories currently indexed.
        _last_rebuild:   ISO timestamp of last full rebuild.
        _memory_meta:    Dict mapping memory_id to metadata (type, tags, etc).
        _search_count:   Total number of searches performed.
        _search_log:     Recent search history.
        _lock:           Thread lock for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the RAG memory with an empty TF-IDF engine."""
        self._tfidf: TFIDFEngine = TFIDFEngine()
        self._indexed_count: int = 0
        self._last_rebuild: Optional[str] = None
        self._memory_meta: Dict[str, Dict[str, Any]] = {}
        self._search_count: int = 0
        self._search_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Try to load persisted index
        self._load_index_if_exists()

        logger.info(
            "RAGMemory initialized (indexed=%d, vocab=%d).",
            self._indexed_count,
            len(self._tfidf._vocabulary),
        )

    # ------------------------------------------------------------------
    # Integration with agent_memory
    # ------------------------------------------------------------------

    def _load_memories(self) -> List[Dict[str, Any]]:
        """
        Lazy-import agent_memory module and load all memories.

        Returns a list of memory dicts with keys:
            id, content, type, tags, created_at, metadata, priority,
            relevance_score, summary, etc.

        Returns:
            List of memory dicts from the agent_memory store.
        """
        try:
            # Lazy import to avoid circular dependencies
            from src.agent_memory import get_memory
            memory_store = get_memory()
        except ImportError:
            logger.warning(
                "Could not import agent_memory module. "
                "Ensure src/agent_memory.py is accessible."
            )
            return []
        except Exception as exc:
            logger.error("Error loading agent_memory: %s", exc)
            return []

        # Access the internal memory store
        memories: List[Dict[str, Any]] = []
        try:
            # AgentMemory stores memories in _memories dict of Memory objects
            with memory_store._lock:
                for mid, mem in memory_store._memories.items():
                    try:
                        mem_dict = mem.to_dict()
                        memories.append(mem_dict)
                    except Exception as exc:
                        logger.warning(
                            "Failed to serialize memory %s: %s", mid, exc
                        )
        except AttributeError:
            # Fallback: try recall with empty query
            logger.debug("Falling back to recall API for memory loading.")
            try:
                from src.agent_memory import MemoryQuery
                query = MemoryQuery(
                    text="",
                    limit=50_000,
                    include_expired=False,
                    include_consolidated=False,
                )
                results = memory_store.recall_sync(query)
                for mem in results:
                    try:
                        memories.append(mem.to_dict())
                    except Exception as exc:
                        logger.warning(
                            "Failed to serialize memory: %s", exc
                        )
            except Exception as exc:
                logger.error("Fallback memory loading failed: %s", exc)

        logger.info("Loaded %d memories from agent_memory store.", len(memories))
        return memories

    def _memory_to_searchable(self, memory: Dict[str, Any]) -> str:
        """
        Convert a memory dict to a searchable text string.

        Combines the memory content, summary, tags, type, and relevant
        metadata fields into a single text block for TF-IDF indexing.

        Args:
            memory: Dict with memory fields (content, tags, type, etc).

        Returns:
            Combined text string for indexing.
        """
        parts: List[str] = []

        # Primary content
        content = memory.get("content", "")
        if content:
            parts.append(content)

        # Summary (if available, adds extra signal)
        summary = memory.get("summary", "")
        if summary:
            parts.append(summary)

        # Tags (very informative for search)
        tags = memory.get("tags", [])
        if tags:
            # Add tags both as-is and space-separated
            parts.append(" ".join(tags))

        # Memory type as a searchable term
        mem_type = memory.get("type", "")
        if mem_type:
            parts.append(mem_type.replace("_", " "))

        # Source module
        source = memory.get("source_module", "")
        if source:
            parts.append(source.replace("_", " "))

        # Metadata values that might be useful for search
        metadata = memory.get("metadata", {})
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, str) and len(value) < 500:
                    parts.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and len(item) < 200:
                            parts.append(item)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_all(self) -> int:
        """
        Load all memories from agent_memory and index them into TF-IDF.

        Performs a full rebuild: clears the existing index, loads all
        memories, tokenizes and indexes each one, then rebuilds IDF
        weights and saves the index to disk.

        Returns:
            Number of memories indexed.
        """
        logger.info("Starting full index rebuild from agent_memory...")
        start_time = time.time()

        # Load all memories
        memories = self._load_memories()

        if not memories:
            logger.warning("No memories found to index.")
            return 0

        # Clear existing index
        self._tfidf.clear()
        self._memory_meta.clear()
        self._indexed_count = 0

        # Index each memory
        indexed = 0
        for mem_dict in memories:
            mem_id = mem_dict.get("id", "")
            if not mem_id:
                continue

            try:
                searchable_text = self._memory_to_searchable(mem_dict)
                if not searchable_text.strip():
                    continue

                self._tfidf.add_document(mem_id, searchable_text)

                # Store metadata for search result enrichment
                self._memory_meta[mem_id] = {
                    "content": mem_dict.get("content", ""),
                    "type": mem_dict.get("type", ""),
                    "tags": mem_dict.get("tags", []),
                    "created_at": mem_dict.get("created_at", ""),
                    "priority": mem_dict.get("priority", "normal"),
                    "relevance_score": mem_dict.get("relevance_score", 1.0),
                    "summary": mem_dict.get("summary", ""),
                    "source_module": mem_dict.get("source_module", ""),
                    "metadata": mem_dict.get("metadata", {}),
                }

                indexed += 1
            except Exception as exc:
                logger.warning("Failed to index memory %s: %s", mem_id, exc)

        # Rebuild IDF with complete corpus
        self._tfidf._rebuild_idf()
        self._tfidf._rebuild_all_vectors()

        self._indexed_count = indexed
        self._last_rebuild = _now_iso()

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "Index rebuild complete: %d memories indexed in %.1f ms "
            "(vocabulary: %d terms).",
            indexed,
            elapsed,
            len(self._tfidf._vocabulary),
        )

        # Save index to disk
        self.save_index()

        return indexed

    def index_all_sync(self) -> int:
        """Synchronous wrapper for index_all()."""
        return _run_sync(self.index_all())

    async def index_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "",
        tags: Optional[List[str]] = None,
        created_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a single memory into the TF-IDF engine.

        Can be called incrementally as new memories are stored, without
        requiring a full rebuild.  Note that IDF weights will be slightly
        stale until the next rebuild_index() call.

        Args:
            memory_id:   Unique ID of the memory.
            content:     Text content of the memory.
            memory_type: Type category (e.g. "app_knowledge").
            tags:        List of tags.
            created_at:  ISO timestamp of creation.
            metadata:    Additional metadata dict.
        """
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        mem_dict = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "tags": tags,
            "created_at": created_at or _now_iso(),
            "metadata": metadata,
        }

        searchable_text = self._memory_to_searchable(mem_dict)
        if not searchable_text.strip():
            logger.debug("Skipping empty memory %s.", memory_id)
            return

        self._tfidf.add_document(memory_id, searchable_text)

        # Store metadata
        with self._lock:
            self._memory_meta[memory_id] = {
                "content": content,
                "type": memory_type,
                "tags": tags,
                "created_at": created_at or _now_iso(),
                "metadata": metadata,
            }
            self._indexed_count = len(self._memory_meta)

        logger.debug("Indexed memory %s (%d tokens).", memory_id, len(searchable_text.split()))

    def index_memory_sync(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "",
        tags: Optional[List[str]] = None,
        created_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronous wrapper for index_memory()."""
        _run_sync(self.index_memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            tags=tags,
            created_at=created_at,
            metadata=metadata,
        ))

    def rebuild_index(self) -> int:
        """
        Force a full re-index from agent_memory.

        Clears the TF-IDF engine and re-indexes all memories from scratch.
        This ensures IDF weights are accurate and the index is consistent.

        Returns:
            Number of memories indexed.
        """
        logger.info("Forcing full index rebuild...")
        return self.index_all_sync()

    # ------------------------------------------------------------------
    # Search — Semantic (TF-IDF cosine similarity)
    # ------------------------------------------------------------------

    async def semantic_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        min_score: float = DEFAULT_MIN_SCORE,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search memories using TF-IDF cosine similarity.

        Args:
            query:       Natural language search query.
            limit:       Maximum results to return.
            min_score:   Minimum cosine similarity threshold.
            memory_type: Filter results to this memory type.
            tags:        Filter results to memories with any of these tags.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        start_time = time.time()

        # Run TF-IDF search
        raw_results = self._tfidf.search(query, limit=limit * 3, min_score=min_score)

        # Convert to SearchResult objects with filtering
        results = self._build_search_results(
            raw_results,
            memory_type=memory_type,
            tags=tags,
            limit=limit,
        )

        elapsed = (time.time() - start_time) * 1000
        self._log_search(query, "semantic", len(results), elapsed, results)

        return results

    def semantic_search_sync(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        min_score: float = DEFAULT_MIN_SCORE,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous wrapper for semantic_search()."""
        return _run_sync(self.semantic_search(
            query=query,
            limit=limit,
            min_score=min_score,
            memory_type=memory_type,
            tags=tags,
        ))

    # ------------------------------------------------------------------
    # Search — Keyword (exact/fuzzy match)
    # ------------------------------------------------------------------

    async def keyword_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search memories using keyword/substring matching.

        Scores documents based on term overlap and exact substring
        matching rather than TF-IDF cosine similarity.

        Args:
            query:       Search query string.
            limit:       Maximum results to return.
            memory_type: Filter by memory type.
            tags:        Filter by tags.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        start_time = time.time()

        raw_results = self._tfidf.keyword_search(query, limit=limit * 3)

        results = self._build_search_results(
            raw_results,
            memory_type=memory_type,
            tags=tags,
            limit=limit,
        )

        elapsed = (time.time() - start_time) * 1000
        self._log_search(query, "keyword", len(results), elapsed, results)

        return results

    def keyword_search_sync(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous wrapper for keyword_search()."""
        return _run_sync(self.keyword_search(
            query=query,
            limit=limit,
            memory_type=memory_type,
            tags=tags,
        ))

    # ------------------------------------------------------------------
    # Search — Hybrid (semantic + keyword combined)
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        min_score: float = DEFAULT_MIN_SCORE,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search using a combination of semantic and keyword strategies.

        Runs both semantic (TF-IDF) and keyword searches, then combines
        scores using a weighted average:

            final_score = semantic_weight * semantic_score
                        + (1 - semantic_weight) * keyword_score

        Args:
            query:           Search query string.
            limit:           Maximum results to return.
            semantic_weight: Weight for semantic scores (0-1, default 0.7).
            min_score:       Minimum combined score threshold.
            memory_type:     Filter by memory type.
            tags:            Filter by tags.

        Returns:
            List of SearchResult objects sorted by combined score descending.
        """
        start_time = time.time()

        keyword_weight = 1.0 - semantic_weight

        # Run both searches with higher limits (we'll filter later)
        fetch_limit = limit * 5
        semantic_raw = self._tfidf.search(query, limit=fetch_limit, min_score=0.01)
        keyword_raw = self._tfidf.keyword_search(query, limit=fetch_limit)

        # Build score maps
        semantic_scores: Dict[str, float] = {
            doc_id: score for doc_id, score in semantic_raw
        }
        keyword_scores: Dict[str, float] = {
            doc_id: score for doc_id, score in keyword_raw
        }

        # Combine all candidate doc_ids
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())

        # Compute combined scores
        combined: List[Tuple[str, float]] = []
        for doc_id in all_doc_ids:
            sem_score = semantic_scores.get(doc_id, 0.0)
            kw_score = keyword_scores.get(doc_id, 0.0)
            final_score = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            if final_score >= min_score:
                combined.append((doc_id, final_score))

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Build search results with filtering
        results = self._build_search_results(
            combined,
            memory_type=memory_type,
            tags=tags,
            limit=limit,
        )

        elapsed = (time.time() - start_time) * 1000
        self._log_search(query, "hybrid", len(results), elapsed, results)

        return results

    def hybrid_search_sync(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        min_score: float = DEFAULT_MIN_SCORE,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous wrapper for hybrid_search()."""
        return _run_sync(self.hybrid_search(
            query=query,
            limit=limit,
            semantic_weight=semantic_weight,
            min_score=min_score,
            memory_type=memory_type,
            tags=tags,
        ))

    # ------------------------------------------------------------------
    # Search result builder (shared)
    # ------------------------------------------------------------------

    def _build_search_results(
        self,
        raw_results: List[Tuple[str, float]],
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[SearchResult]:
        """
        Convert raw (doc_id, score) pairs to SearchResult objects.

        Applies memory_type and tag filters, enriches with metadata,
        and limits output count.

        Args:
            raw_results: List of (doc_id, score) from the TF-IDF engine.
            memory_type: Optional type filter (exact match).
            tags:        Optional tag filter (any match).
            limit:       Maximum results.

        Returns:
            Filtered, enriched list of SearchResult objects.
        """
        results: List[SearchResult] = []

        for doc_id, score in raw_results:
            if len(results) >= limit:
                break

            meta = self._memory_meta.get(doc_id, {})

            # Apply memory type filter
            if memory_type is not None:
                doc_type = meta.get("type", "")
                if doc_type != memory_type:
                    continue

            # Apply tag filter (any match)
            if tags:
                doc_tags = meta.get("tags", [])
                if not any(t in doc_tags for t in tags):
                    continue

            result = SearchResult(
                memory_id=doc_id,
                content=meta.get("content", ""),
                score=round(score, 4),
                memory_type=meta.get("type", ""),
                tags=meta.get("tags", []),
                created_at=meta.get("created_at", ""),
                metadata=meta.get("metadata", {}),
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Context Building (the main RAG purpose)
    # ------------------------------------------------------------------

    async def build_context(
        self,
        query: str,
        max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 20,
        min_score: float = DEFAULT_MIN_SCORE,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> RAGContext:
        """
        Build LLM-ready context from relevant memories.

        Searches for memories matching the query, then formats the
        results into a context string suitable for injection into an
        LLM system prompt or user message.

        The context string is formatted as a numbered list with metadata
        headers and content bodies, truncated to fit within max_tokens.

        Args:
            query:       Natural language query describing the task.
            max_tokens:  Maximum estimated tokens for the context string.
            strategy:    Search strategy to use.
            limit:       Maximum memories to retrieve.
            min_score:   Minimum relevance score.
            memory_type: Filter by memory type.
            tags:        Filter by tags.

        Returns:
            RAGContext with formatted context string and metadata.
        """
        start_time = time.time()

        # Run the appropriate search strategy
        if strategy == SearchStrategy.SEMANTIC:
            results = await self.semantic_search(
                query, limit=limit, min_score=min_score,
                memory_type=memory_type, tags=tags,
            )
        elif strategy == SearchStrategy.KEYWORD:
            results = await self.keyword_search(
                query, limit=limit,
                memory_type=memory_type, tags=tags,
            )
        else:  # HYBRID
            results = await self.hybrid_search(
                query, limit=limit, min_score=min_score,
                memory_type=memory_type, tags=tags,
            )

        # Format context string with token budget
        context_string = self._format_context(results, max_tokens)
        token_estimate = self._estimate_tokens(context_string)

        elapsed = (time.time() - start_time) * 1000

        return RAGContext(
            query=query,
            results=results,
            context_string=context_string,
            token_estimate=token_estimate,
            strategy_used=strategy,
            search_time_ms=round(elapsed, 2),
        )

    def build_context_sync(
        self,
        query: str,
        max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 20,
        min_score: float = DEFAULT_MIN_SCORE,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> RAGContext:
        """Synchronous wrapper for build_context()."""
        return _run_sync(self.build_context(
            query=query,
            max_tokens=max_tokens,
            strategy=strategy,
            limit=limit,
            min_score=min_score,
            memory_type=memory_type,
            tags=tags,
        ))

    async def conversation_context(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
    ) -> RAGContext:
        """
        Build context from a multi-turn conversation history.

        Extracts key terms from the recent messages (prioritizing the
        last 3 user messages) and builds a combined query for context
        retrieval.

        Args:
            messages:   List of message dicts with "role" and "content" keys.
            max_tokens: Maximum tokens for the context string.
            strategy:   Search strategy to use.

        Returns:
            RAGContext built from the conversation's semantic content.
        """
        if not messages:
            return RAGContext(
                query="",
                results=[],
                context_string="",
                token_estimate=0,
                strategy_used=strategy,
                search_time_ms=0.0,
            )

        # Extract user messages, prioritize recent ones
        user_messages: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                user_messages.append(content)

        # Take the last 3 user messages for context
        recent_messages = user_messages[-3:] if len(user_messages) > 3 else user_messages

        # Build combined query from recent messages
        # Weight the most recent message higher by repeating it
        if recent_messages:
            query_parts = []
            for i, msg in enumerate(recent_messages):
                # Truncate very long messages
                truncated = msg[:500] if len(msg) > 500 else msg
                # Repeat the most recent message for higher weight
                if i == len(recent_messages) - 1:
                    query_parts.extend([truncated, truncated])
                else:
                    query_parts.append(truncated)
            combined_query = " ".join(query_parts)
        else:
            # Fall back to the last message of any role
            combined_query = messages[-1].get("content", "")[:500]

        return await self.build_context(
            query=combined_query,
            max_tokens=max_tokens,
            strategy=strategy,
        )

    def conversation_context_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
    ) -> RAGContext:
        """Synchronous wrapper for conversation_context()."""
        return _run_sync(self.conversation_context(
            messages=messages,
            max_tokens=max_tokens,
            strategy=strategy,
        ))

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def _format_context(
        self,
        results: List[SearchResult],
        max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    ) -> str:
        """
        Format search results into a context string for LLM prompts.

        Produces a structured text block like:

            --- Relevant Memory [1/5] (score: 0.87) ---
            Type: app_knowledge | Tags: instagram, login, 2fa
            Created: 2025-01-15T10:30:00+00:00

            Instagram login requires 2FA code from SMS. The code is
            sent to the registered phone number and expires after 10
            minutes.

            ---

        Results are included in order until the token budget is exhausted.

        Args:
            results:    List of SearchResult objects to format.
            max_tokens: Maximum estimated tokens for the output.

        Returns:
            Formatted context string.
        """
        if not results:
            return ""

        max_chars = max_tokens * CHARS_PER_TOKEN
        parts: List[str] = []
        current_chars = 0
        total = len(results)

        header = f"=== Retrieved Context ({total} relevant memories) ===\n"
        parts.append(header)
        current_chars += len(header)

        included = 0
        for i, result in enumerate(results):
            # Build the entry
            entry_lines: List[str] = []

            # Header line
            entry_lines.append(
                f"--- Relevant Memory [{i + 1}/{total}] "
                f"(score: {result.score:.2f}) ---"
            )

            # Metadata line
            meta_parts = []
            if result.memory_type:
                meta_parts.append(f"Type: {result.memory_type}")
            if result.tags:
                meta_parts.append(f"Tags: {', '.join(result.tags)}")
            if meta_parts:
                entry_lines.append(" | ".join(meta_parts))

            # Timestamp
            if result.created_at:
                entry_lines.append(f"Created: {result.created_at}")

            # Content
            entry_lines.append("")
            content = result.content
            # Truncate very long content
            remaining_budget = max_chars - current_chars - 200  # reserve for framing
            if remaining_budget <= 0:
                break
            if len(content) > remaining_budget:
                content = content[:remaining_budget - 3] + "..."
            entry_lines.append(content)
            entry_lines.append("")

            entry = "\n".join(entry_lines)

            # Check token budget
            if current_chars + len(entry) > max_chars and included > 0:
                # Add a note about truncation
                parts.append(
                    f"\n[... {total - included} more memories omitted "
                    f"due to token budget ...]\n"
                )
                break

            parts.append(entry)
            current_chars += len(entry)
            included += 1

        parts.append("=== End Retrieved Context ===")

        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimation.

        Uses the approximation: 1 token ~= 4 characters.
        This is intentionally conservative (overestimates) to avoid
        exceeding context windows.

        Args:
            text: Text to estimate token count for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return max(1, len(text) // CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive statistics about the RAG memory index.

        Returns:
            Dict with index stats, search stats, and TF-IDF engine stats.
        """
        tfidf_stats = self._tfidf.get_stats()

        # Compute memory type distribution
        type_counts: Dict[str, int] = defaultdict(int)
        for meta in self._memory_meta.values():
            mem_type = meta.get("type", "unknown")
            type_counts[mem_type] += 1

        # Compute tag distribution
        tag_counts: Dict[str, int] = defaultdict(int)
        for meta in self._memory_meta.values():
            for tag in meta.get("tags", []):
                tag_counts[tag] += 1

        # Sort tag counts
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "indexed_memories": self._indexed_count,
            "last_rebuild": self._last_rebuild,
            "search_count": self._search_count,
            "tfidf_engine": tfidf_stats,
            "memory_type_distribution": dict(type_counts),
            "top_tags": top_tags,
            "search_log_size": len(self._search_log),
            "data_dir": str(RAG_DATA_DIR),
        }

    def get_top_terms(self, limit: int = 50) -> List[Tuple[str, float]]:
        """
        Return the highest-IDF terms in the vocabulary.

        These are the most discriminating terms — rare words that
        appear in few documents and thus carry the most information
        for distinguishing between memories.

        Args:
            limit: Maximum terms to return.

        Returns:
            List of (term, idf_weight) sorted by IDF descending.
        """
        return self._tfidf.get_top_terms(limit)

    def get_similar_memories(
        self,
        memory_id: str,
        limit: int = 5,
    ) -> List[SearchResult]:
        """
        Find memories similar to a given memory.

        Uses the indexed document's text as a search query to find
        other memories with high cosine similarity.

        Args:
            memory_id: ID of the reference memory.
            limit:     Maximum similar memories to return.

        Returns:
            List of SearchResult objects (excluding the query memory itself).
        """
        # Get the raw text of the reference memory
        raw_text = self._tfidf._raw_texts.get(memory_id, "")
        if not raw_text:
            logger.warning("Memory %s not found in index.", memory_id)
            return []

        # Search using the memory's text as query
        raw_results = self._tfidf.search(raw_text, limit=limit + 1, min_score=0.01)

        # Filter out the query memory itself
        filtered = [(doc_id, score) for doc_id, score in raw_results if doc_id != memory_id]

        return self._build_search_results(filtered, limit=limit)

    def get_similar_memories_sync(
        self,
        memory_id: str,
        limit: int = 5,
    ) -> List[SearchResult]:
        """Synchronous version of get_similar_memories (not async, no wrapper needed)."""
        return self.get_similar_memories(memory_id, limit)

    # ------------------------------------------------------------------
    # Search logging
    # ------------------------------------------------------------------

    def _log_search(
        self,
        query: str,
        strategy: str,
        result_count: int,
        elapsed_ms: float,
        results: List[SearchResult],
    ) -> None:
        """
        Log a search operation for analytics.

        Args:
            query:        The search query.
            strategy:     Search strategy used.
            result_count: Number of results returned.
            elapsed_ms:   Search time in milliseconds.
            results:      The search results.
        """
        with self._lock:
            self._search_count += 1

            entry = SearchLogEntry(
                query=query[:200],  # truncate long queries
                strategy=strategy,
                result_count=result_count,
                top_score=results[0].score if results else 0.0,
                search_time_ms=round(elapsed_ms, 2),
                timestamp=_now_iso(),
            )

            self._search_log.append(entry.to_dict())

            # Trim log if too large
            if len(self._search_log) > MAX_SEARCH_LOG_ENTRIES:
                self._search_log = self._search_log[-MAX_SEARCH_LOG_ENTRIES:]

    def get_search_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return recent search history.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of search log entry dicts, most recent first.
        """
        entries = list(reversed(self._search_log))
        return entries[:limit]

    # ------------------------------------------------------------------
    # Persistence — Index Save/Load
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        """
        Persist the TF-IDF index and metadata to disk.

        Saves three files atomically:
            - tfidf_index.json: Full TF-IDF engine state.
            - index_meta.json:  Metadata (counts, timestamps, memory_meta).
            - search_log.json:  Recent search history.
        """
        logger.debug("Saving RAG index to disk...")

        try:
            # Save TF-IDF engine state
            tfidf_state = self._tfidf.to_dict()
            _save_json(TFIDF_INDEX_FILE, tfidf_state)

            # Save index metadata
            meta = {
                "indexed_count": self._indexed_count,
                "last_rebuild": self._last_rebuild,
                "search_count": self._search_count,
                "memory_meta": self._memory_meta,
                "saved_at": _now_iso(),
            }
            _save_json(INDEX_META_FILE, meta)

            # Save search log
            _save_json(SEARCH_LOG_FILE, self._search_log)

            logger.info(
                "RAG index saved: %d memories, %d vocab terms.",
                self._indexed_count,
                len(self._tfidf._vocabulary),
            )
        except Exception as exc:
            logger.error("Failed to save RAG index: %s", exc)

    def load_index(self) -> bool:
        """
        Load the TF-IDF index and metadata from disk.

        Returns:
            True if index was loaded successfully, False otherwise.
        """
        logger.debug("Loading RAG index from disk...")

        try:
            # Load TF-IDF engine state
            tfidf_data = _load_json(TFIDF_INDEX_FILE, default={})
            if not tfidf_data:
                logger.info("No persisted TF-IDF index found.")
                return False

            self._tfidf = TFIDFEngine.from_dict(tfidf_data)

            # Load index metadata
            meta = _load_json(INDEX_META_FILE, default={})
            self._indexed_count = meta.get("indexed_count", 0)
            self._last_rebuild = meta.get("last_rebuild")
            self._search_count = meta.get("search_count", 0)
            self._memory_meta = meta.get("memory_meta", {})

            # Load search log
            self._search_log = _load_json(SEARCH_LOG_FILE, default=[])
            if not isinstance(self._search_log, list):
                self._search_log = []

            logger.info(
                "RAG index loaded: %d memories, %d vocab terms.",
                self._indexed_count,
                len(self._tfidf._vocabulary),
            )
            return True

        except Exception as exc:
            logger.error("Failed to load RAG index: %s", exc)
            return False

    def _load_index_if_exists(self) -> None:
        """Try to load a persisted index, silently fail if not found."""
        if TFIDF_INDEX_FILE.exists() and INDEX_META_FILE.exists():
            self.load_index()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_index(self) -> None:
        """Clear the entire RAG index from memory and disk."""
        self._tfidf.clear()
        self._memory_meta.clear()
        self._indexed_count = 0
        self._last_rebuild = None
        self._search_count = 0
        self._search_log.clear()

        # Remove persisted files
        for filepath in [TFIDF_INDEX_FILE, INDEX_META_FILE, SEARCH_LOG_FILE]:
            try:
                if filepath.exists():
                    filepath.unlink()
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", filepath, exc)

        logger.info("RAG index cleared.")

    def is_indexed(self) -> bool:
        """Return True if the index contains any documents."""
        return self._indexed_count > 0

    def index_age_hours(self) -> Optional[float]:
        """
        Return the age of the current index in hours.

        Returns:
            Hours since last rebuild, or None if never built.
        """
        if not self._last_rebuild:
            return None
        dt = _parse_iso(self._last_rebuild)
        if dt is None:
            return None
        delta = _now_utc() - dt
        return max(delta.total_seconds() / 3600.0, 0.0)

    def needs_rebuild(self, max_age_hours: float = 24.0) -> bool:
        """
        Check if the index is stale and needs a rebuild.

        Args:
            max_age_hours: Maximum age before the index is considered stale.

        Returns:
            True if the index is empty or older than max_age_hours.
        """
        if not self.is_indexed():
            return True
        age = self.index_age_hours()
        if age is None:
            return True
        return age > max_age_hours


# ===================================================================
# SINGLETON
# ===================================================================

_rag_memory: Optional[RAGMemory] = None


def get_rag_memory() -> RAGMemory:
    """
    Get the global RAGMemory singleton.

    Creates the instance on first call, loading any persisted index
    from disk.

    Returns:
        The global RAGMemory instance.
    """
    global _rag_memory
    if _rag_memory is None:
        _rag_memory = RAGMemory()
    return _rag_memory


# ===================================================================
# CLI FORMATTING
# ===================================================================

def _format_table(
    headers: List[str],
    rows: List[List[str]],
    max_col_width: int = 50,
) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    # Truncate long values
    truncated_rows: List[List[str]] = []
    for row in rows:
        truncated_rows.append([
            val[: max_col_width - 3] + "..." if len(val) > max_col_width else val
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


def _format_search_result_detail(result: SearchResult) -> str:
    """Format a single search result for detailed display."""
    lines = [
        f"  Memory ID:   {result.memory_id}",
        f"  Score:       {result.score:.4f}",
        f"  Type:        {result.memory_type}",
        f"  Tags:        {', '.join(result.tags) if result.tags else 'none'}",
        f"  Created:     {result.created_at}",
        f"  Content:",
    ]

    content = result.content
    if len(content) > 500:
        content = content[:497] + "..."
    for line in content.split("\n"):
        lines.append(f"    {line}")

    if result.metadata:
        lines.append(f"  Metadata:    {json.dumps(result.metadata, default=str)[:200]}")

    return "\n".join(lines)


def _format_context_detail(ctx: RAGContext) -> str:
    """Format a RAGContext for CLI display."""
    lines = [
        f"  Query:          {ctx.query[:100]}",
        f"  Strategy:       {ctx.strategy_used.value}",
        f"  Results:        {len(ctx.results)}",
        f"  Token Estimate: {ctx.token_estimate}",
        f"  Search Time:    {ctx.search_time_ms:.1f} ms",
        "",
        "  Context String:",
        "  " + "-" * 60,
    ]

    for line in ctx.context_string.split("\n"):
        lines.append(f"  {line}")

    lines.append("  " + "-" * 60)
    return "\n".join(lines)


# ===================================================================
# CLI COMMANDS
# ===================================================================

def _cmd_index(args: argparse.Namespace) -> None:
    """Rebuild the RAG index from agent_memory."""
    rag = get_rag_memory()
    print("\n  Rebuilding RAG index from agent_memory...\n")

    count = rag.rebuild_index()

    stats = rag.get_stats()
    print(f"  Index rebuilt successfully.")
    print(f"  Memories indexed:  {count}")
    print(f"  Vocabulary size:   {stats['tfidf_engine']['vocabulary_size']}")
    print(f"  Total tokens:      {stats['tfidf_engine']['total_tokens']}")
    print(f"  Avg doc length:    {stats['tfidf_engine']['avg_document_length']}")
    print(f"  Last rebuild:      {rag._last_rebuild}")
    print()


def _cmd_search(args: argparse.Namespace) -> None:
    """Search the RAG index."""
    rag = get_rag_memory()

    if not rag.is_indexed():
        print("\n  Index is empty. Run 'index' first.\n")
        return

    query = args.query
    if not query:
        print("\n  Error: --query is required.\n")
        return

    strategy_str = args.strategy.lower()
    try:
        strategy = SearchStrategy(strategy_str)
    except ValueError:
        print(f"\n  Invalid strategy: {strategy_str}")
        print(f"  Valid: {', '.join(s.value for s in SearchStrategy)}\n")
        return

    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    print(f"\n  Searching: \"{query}\"")
    print(f"  Strategy:  {strategy.value}")
    print(f"  Limit:     {args.limit}")
    if args.type:
        print(f"  Type:      {args.type}")
    if tags:
        print(f"  Tags:      {', '.join(tags)}")
    print()

    if strategy == SearchStrategy.SEMANTIC:
        results = rag.semantic_search_sync(
            query=query,
            limit=args.limit,
            min_score=args.min_score,
            memory_type=args.type,
            tags=tags,
        )
    elif strategy == SearchStrategy.KEYWORD:
        results = rag.keyword_search_sync(
            query=query,
            limit=args.limit,
            memory_type=args.type,
            tags=tags,
        )
    else:
        results = rag.hybrid_search_sync(
            query=query,
            limit=args.limit,
            min_score=args.min_score,
            memory_type=args.type,
            tags=tags,
        )

    if not results:
        print("  No results found.\n")
        return

    if args.detail and len(results) <= 3:
        for i, result in enumerate(results):
            print(f"\n  --- Result {i + 1}/{len(results)} ---")
            print(_format_search_result_detail(result))
        print()
    else:
        rows = []
        for result in results:
            content_preview = result.content[:60].replace("\n", " ")
            if len(result.content) > 60:
                content_preview += "..."
            tags_str = ", ".join(result.tags[:3])
            if len(result.tags) > 3:
                tags_str += f" (+{len(result.tags) - 3})"
            rows.append([
                result.memory_id[:12] + "...",
                f"{result.score:.3f}",
                result.memory_type,
                tags_str,
                content_preview,
            ])

        print(_format_table(
            ["ID", "Score", "Type", "Tags", "Content"],
            rows,
        ))
        print(f"\n  {len(results)} result(s) found.\n")


def _cmd_context(args: argparse.Namespace) -> None:
    """Build LLM context from the RAG index."""
    rag = get_rag_memory()

    if not rag.is_indexed():
        print("\n  Index is empty. Run 'index' first.\n")
        return

    query = args.query
    if not query:
        print("\n  Error: --query is required.\n")
        return

    strategy_str = args.strategy.lower()
    try:
        strategy = SearchStrategy(strategy_str)
    except ValueError:
        print(f"\n  Invalid strategy: {strategy_str}")
        print(f"  Valid: {', '.join(s.value for s in SearchStrategy)}\n")
        return

    print(f"\n  Building context for: \"{query}\"")
    print(f"  Strategy:   {strategy.value}")
    print(f"  Max tokens: {args.max_tokens}")
    print()

    ctx = rag.build_context_sync(
        query=query,
        max_tokens=args.max_tokens,
        strategy=strategy,
    )

    print(_format_context_detail(ctx))
    print()


def _cmd_similar(args: argparse.Namespace) -> None:
    """Find memories similar to a given memory."""
    rag = get_rag_memory()

    if not rag.is_indexed():
        print("\n  Index is empty. Run 'index' first.\n")
        return

    memory_id = args.memory_id

    # Try to match partial IDs
    if len(memory_id) < 36:
        # Find matching IDs
        matches = [
            mid for mid in rag._memory_meta.keys()
            if mid.startswith(memory_id)
        ]
        if len(matches) == 0:
            print(f"\n  No memory found matching: {memory_id}\n")
            return
        elif len(matches) == 1:
            memory_id = matches[0]
            print(f"\n  Matched to: {memory_id}")
        else:
            print(f"\n  Ambiguous ID prefix. Matches:")
            for mid in matches[:10]:
                print(f"    {mid}")
            if len(matches) > 10:
                print(f"    ... and {len(matches) - 10} more")
            print()
            return

    results = rag.get_similar_memories(memory_id, limit=args.limit)

    if not results:
        print(f"\n  No similar memories found for {memory_id}.\n")
        return

    # Show reference memory
    ref_meta = rag._memory_meta.get(memory_id, {})
    print(f"\n  Reference memory: {memory_id}")
    print(f"  Type: {ref_meta.get('type', '?')}")
    ref_content = ref_meta.get("content", "")[:80]
    print(f"  Content: {ref_content}...")
    print()

    rows = []
    for result in results:
        content_preview = result.content[:60].replace("\n", " ")
        if len(result.content) > 60:
            content_preview += "..."
        rows.append([
            result.memory_id[:12] + "...",
            f"{result.score:.3f}",
            result.memory_type,
            content_preview,
        ])

    print("  Similar memories:")
    print(_format_table(["ID", "Similarity", "Type", "Content"], rows))
    print()


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show RAG index statistics."""
    rag = get_rag_memory()
    stats = rag.get_stats()

    print("\n  RAG Memory Index Statistics")
    print("  " + "=" * 50)
    print(f"  Indexed Memories:     {stats['indexed_memories']}")
    print(f"  Last Rebuild:         {stats['last_rebuild'] or 'never'}")
    print(f"  Total Searches:       {stats['search_count']}")
    print(f"  Search Log Size:      {stats['search_log_size']}")
    print(f"  Data Directory:       {stats['data_dir']}")
    print()

    tfidf = stats["tfidf_engine"]
    print("  TF-IDF Engine")
    print("  " + "-" * 40)
    print(f"  Document Count:       {tfidf['document_count']}")
    print(f"  Vocabulary Size:      {tfidf['vocabulary_size']}")
    print(f"  Total Tokens:         {tfidf['total_tokens']}")
    print(f"  Avg Doc Length:       {tfidf['avg_document_length']}")
    print(f"  IDF Terms:            {tfidf['idf_terms']}")
    print(f"  Stemmer Cache:        {tfidf['stemmer_cache_size']}")
    print(f"  Needs Rebuild:        {tfidf['needs_rebuild']}")
    print()

    # Memory type distribution
    type_dist = stats.get("memory_type_distribution", {})
    if type_dist:
        print("  Memory Type Distribution")
        print("  " + "-" * 40)
        sorted_types = sorted(type_dist.items(), key=lambda x: x[1], reverse=True)
        for mem_type, count in sorted_types:
            bar = "#" * min(count, 40)
            print(f"  {mem_type:<20} {count:>5}  {bar}")
        print()

    # Top tags
    top_tags = stats.get("top_tags", [])
    if top_tags:
        print("  Top Tags")
        print("  " + "-" * 40)
        for tag, count in top_tags[:15]:
            bar = "#" * min(count, 30)
            print(f"  {tag:<25} {count:>4}  {bar}")
        print()

    # Index age
    age = rag.index_age_hours()
    if age is not None:
        if age < 1:
            age_str = f"{age * 60:.0f} minutes"
        elif age < 24:
            age_str = f"{age:.1f} hours"
        else:
            age_str = f"{age / 24:.1f} days"
        print(f"  Index Age:            {age_str}")
        print(f"  Needs Rebuild:        {rag.needs_rebuild()}")
    print()

    # Recent searches
    history = rag.get_search_history(limit=5)
    if history:
        print("  Recent Searches")
        print("  " + "-" * 40)
        rows = []
        for entry in history:
            rows.append([
                entry.get("query", "")[:30],
                entry.get("strategy", ""),
                str(entry.get("result_count", 0)),
                f"{entry.get('top_score', 0.0):.3f}",
                f"{entry.get('search_time_ms', 0.0):.1f}ms",
            ])
        print(_format_table(
            ["Query", "Strategy", "Results", "Top Score", "Time"],
            rows,
        ))
        print()


def _cmd_terms(args: argparse.Namespace) -> None:
    """Show top IDF terms in the vocabulary."""
    rag = get_rag_memory()

    if not rag.is_indexed():
        print("\n  Index is empty. Run 'index' first.\n")
        return

    terms = rag.get_top_terms(limit=args.limit)

    if not terms:
        print("\n  No terms in vocabulary.\n")
        return

    print(f"\n  Top {len(terms)} IDF Terms (most discriminating)")
    print("  " + "=" * 50)

    rows = []
    for term, idf_weight in terms:
        bar = "#" * min(int(idf_weight * 5), 40)
        rows.append([term, f"{idf_weight:.4f}", bar])

    print(_format_table(["Term", "IDF Weight", "Relative Weight"], rows))
    print(f"\n  Total vocabulary: {len(rag._tfidf._vocabulary)} terms\n")


def _cmd_history(args: argparse.Namespace) -> None:
    """Show recent search history."""
    rag = get_rag_memory()

    history = rag.get_search_history(limit=args.limit)

    if not history:
        print("\n  No search history.\n")
        return

    print(f"\n  Search History (most recent {len(history)} entries)")
    print("  " + "=" * 50)

    rows = []
    for entry in history:
        rows.append([
            entry.get("timestamp", "")[:19],
            entry.get("query", "")[:40],
            entry.get("strategy", ""),
            str(entry.get("result_count", 0)),
            f"{entry.get('top_score', 0.0):.3f}",
            f"{entry.get('search_time_ms', 0.0):.1f}ms",
        ])

    print(_format_table(
        ["Timestamp", "Query", "Strategy", "Results", "Top Score", "Time"],
        rows,
        max_col_width=42,
    ))
    print()


def _cmd_clear(args: argparse.Namespace) -> None:
    """Clear the RAG index."""
    rag = get_rag_memory()

    if not args.confirm:
        print("\n  This will delete the RAG index. Use --confirm to proceed.\n")
        return

    rag.clear_index()
    print("\n  RAG index cleared.\n")


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the RAG memory module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="rag_memory",
        description="OpenClaw Empire RAG Memory -- TF-IDF Semantic Search over Agent Memory",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- index ----
    sp_index = subparsers.add_parser(
        "index",
        help="Rebuild the RAG index from agent_memory",
    )
    sp_index.set_defaults(func=_cmd_index)

    # ---- search ----
    sp_search = subparsers.add_parser(
        "search",
        help="Search the RAG index",
    )
    sp_search.add_argument(
        "--query", "-q", type=str, required=True,
        help="Search query text",
    )
    sp_search.add_argument(
        "--strategy", "-s", type=str, default="hybrid",
        help=f"Search strategy ({', '.join(s.value for s in SearchStrategy)})",
    )
    sp_search.add_argument(
        "--limit", "-l", type=int, default=DEFAULT_SEARCH_LIMIT,
        help=f"Max results (default: {DEFAULT_SEARCH_LIMIT})",
    )
    sp_search.add_argument(
        "--min-score", type=float, default=DEFAULT_MIN_SCORE,
        help=f"Minimum score threshold (default: {DEFAULT_MIN_SCORE})",
    )
    sp_search.add_argument(
        "--type", "-t", type=str, default=None,
        help="Filter by memory type",
    )
    sp_search.add_argument(
        "--tags", type=str, default=None,
        help="Comma-separated tag filter (any match)",
    )
    sp_search.add_argument(
        "--detail", "-d", action="store_true",
        help="Show detailed view (for <= 3 results)",
    )
    sp_search.set_defaults(func=_cmd_search)

    # ---- context ----
    sp_context = subparsers.add_parser(
        "context",
        help="Build LLM context from relevant memories",
    )
    sp_context.add_argument(
        "--query", "-q", type=str, required=True,
        help="Context query text",
    )
    sp_context.add_argument(
        "--max-tokens", type=int, default=DEFAULT_CONTEXT_MAX_TOKENS,
        help=f"Maximum context tokens (default: {DEFAULT_CONTEXT_MAX_TOKENS})",
    )
    sp_context.add_argument(
        "--strategy", "-s", type=str, default="hybrid",
        help=f"Search strategy ({', '.join(s.value for s in SearchStrategy)})",
    )
    sp_context.set_defaults(func=_cmd_context)

    # ---- similar ----
    sp_similar = subparsers.add_parser(
        "similar",
        help="Find memories similar to a given memory",
    )
    sp_similar.add_argument(
        "--memory-id", "-m", type=str, required=True,
        help="Memory ID (full UUID or prefix)",
    )
    sp_similar.add_argument(
        "--limit", "-l", type=int, default=5,
        help="Max similar memories (default: 5)",
    )
    sp_similar.set_defaults(func=_cmd_similar)

    # ---- stats ----
    sp_stats = subparsers.add_parser(
        "stats",
        help="Show RAG index statistics",
    )
    sp_stats.set_defaults(func=_cmd_stats)

    # ---- terms ----
    sp_terms = subparsers.add_parser(
        "terms",
        help="Show top IDF terms (most discriminating vocabulary)",
    )
    sp_terms.add_argument(
        "--limit", "-l", type=int, default=50,
        help="Max terms to show (default: 50)",
    )
    sp_terms.set_defaults(func=_cmd_terms)

    # ---- history ----
    sp_history = subparsers.add_parser(
        "history",
        help="Show recent search history",
    )
    sp_history.add_argument(
        "--limit", "-l", type=int, default=50,
        help="Max entries to show (default: 50)",
    )
    sp_history.set_defaults(func=_cmd_history)

    # ---- clear ----
    sp_clear = subparsers.add_parser(
        "clear",
        help="Clear the RAG index (requires --confirm)",
    )
    sp_clear.add_argument(
        "--confirm", action="store_true",
        help="Confirm index deletion",
    )
    sp_clear.set_defaults(func=_cmd_clear)

    # Parse and dispatch
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
