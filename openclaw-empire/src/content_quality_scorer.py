"""
Content Quality Scorer -- OpenClaw Empire Edition
===================================================

Scores article content across 7 quality dimensions: readability, E-E-A-T
signals, SEO health, structure, originality, engagement, and voice match.
Produces a composite grade (A+ through F) and enforces quality gates before
content is published to any of the 16 empire sites.

Quality Dimensions:
    1. Readability   -- Flesch-Kincaid grade level, sentence length, passive voice
    2. E-E-A-T       -- Experience, Expertise, Authoritativeness, Trust markers
    3. SEO           -- Keyword placement, heading structure, meta estimates
    4. Structure     -- Heading hierarchy, paragraph distribution, lists, intro/conclusion
    5. Originality   -- Shingle-based fingerprinting, Jaccard similarity, flagged phrases
    6. Engagement    -- Hook strength, CTA presence, questions, story elements
    7. Voice Match   -- Brand voice adherence via lazy BrandVoiceEngine import

All scoring is pure Python -- no NLTK, no spaCy, no external NLP libraries.
Readability uses vowel-group syllable counting and the Flesch-Kincaid formula.

Usage:
    from src.content_quality_scorer import get_scorer

    scorer = get_scorer()
    report = await scorer.score(
        content="<h2>Introduction</h2><p>Your article text...</p>",
        title="Full Moon Ritual Guide",
        site_id="witchcraft",
        keywords=["full moon", "ritual"],
    )

    passed, reasons = scorer.check_gate(report)
    if not passed:
        print(f"Quality gate FAILED: {reasons}")

CLI:
    python -m src.content_quality_scorer score --content article.html --title "Moon Ritual" --site witchcraft
    python -m src.content_quality_scorer score --stdin --title "Moon Ritual" --site witchcraft
    python -m src.content_quality_scorer history --site witchcraft --limit 20
    python -m src.content_quality_scorer trend --site witchcraft --days 30
    python -m src.content_quality_scorer stats
    python -m src.content_quality_scorer threshold --score 7.0
    python -m src.content_quality_scorer failing --days 7
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("content_quality_scorer")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
QUALITY_DATA_DIR = BASE_DIR / "data" / "quality"
CORPUS_DIR = QUALITY_DATA_DIR / "corpus"
HISTORY_FILE = QUALITY_DATA_DIR / "history.json"
CONFIG_FILE = QUALITY_DATA_DIR / "config.json"

# Ensure data directories exist on import
QUALITY_DATA_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# Reading speed for estimated reading time (words per minute)
READING_SPEED_WPM = 238

# Shingle size for fingerprinting (number of consecutive words per shingle)
SHINGLE_SIZE = 5

# Maximum articles to keep in history
MAX_HISTORY = 2000

# Default quality gate threshold
DEFAULT_THRESHOLD = 6.0

# Default dimension weights (sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "readability": 0.15,
    "eeat": 0.15,
    "seo": 0.15,
    "structure": 0.15,
    "originality": 0.15,
    "engagement": 0.10,
    "voice_match": 0.15,
}

# Valid site IDs across the empire
VALID_SITE_IDS = (
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
)

# Passive voice auxiliary verbs for detection
PASSIVE_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being",
    "has been", "have been", "had been", "will be",
    "is being", "are being", "was being", "were being",
}

# Common flagged phrases that signal AI-generated or low-quality content
FLAGGED_PHRASES = [
    "in conclusion",
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "it's worth noting",
    "in today's world",
    "in this day and age",
    "at the end of the day",
    "without further ado",
    "in this article we will",
    "in this blog post",
    "let's dive in",
    "let's get started",
    "buckle up",
    "stay tuned",
    "as we all know",
    "needless to say",
    "it goes without saying",
    "last but not least",
    "first and foremost",
    "each and every",
    "few and far between",
    "the fact of the matter",
    "when it comes to",
    "at this point in time",
    "for all intents and purposes",
    "delve into",
    "delve deeper",
    "tapestry of",
    "rich tapestry",
    "landscape of",
    "in the realm of",
    "navigate the",
    "navigating the",
    "unlock the",
    "unlocking the",
    "embark on",
    "embarking on",
    "harness the power",
    "harnessing the power",
    "game-changer",
    "game changer",
    "cutting-edge",
    "leverage the",
    "leveraging the",
    "revolutionize",
    "revolutionizing",
    "comprehensive guide",
    "ultimate guide",
]

# E-E-A-T signal words and phrases
EXPERIENCE_SIGNALS = [
    "i've", "i have", "in my experience", "i found", "i discovered",
    "when i", "i personally", "i recommend", "my favorite", "i tested",
    "i tried", "i used", "years of", "i learned", "from my",
    "i noticed", "i realized", "i observed", "i've been",
    "in my practice", "my approach", "i suggest",
]

EXPERTISE_SIGNALS = [
    "research shows", "studies show", "according to", "data shows",
    "statistics", "percent", "%", "published in", "journal",
    "peer-reviewed", "meta-analysis", "clinical", "evidence-based",
    "scientifically", "proven", "survey of", "findings",
    "methodology", "systematic", "empirical", "quantitative",
]

AUTHORITY_SIGNALS = [
    "certified", "licensed", "board-certified", "accredited",
    "credential", "degree", "university", "published author",
    "award-winning", "recognized", "expert", "specialist",
    "professional", "practitioner", "trained in",
]

TRUST_SIGNALS = [
    "disclaimer", "consult", "seek professional", "not medical advice",
    "not financial advice", "speak with", "talk to your",
    "source:", "sources:", "reference:", "references:",
    "cited", "verified", "fact-checked", "updated",
    "affiliate", "disclosure", "transparency",
]

# CTA patterns
CTA_PATTERNS = [
    r"sign\s+up", r"subscribe", r"download", r"get\s+started",
    r"try\s+it", r"join\s+", r"click\s+here", r"learn\s+more",
    r"read\s+more", r"check\s+out", r"grab\s+your", r"claim\s+your",
    r"don't\s+miss", r"start\s+your", r"begin\s+your",
    r"take\s+the\s+first\s+step", r"ready\s+to",
    r"share\s+this", r"leave\s+a\s+comment", r"tell\s+us",
    r"what\s+do\s+you\s+think", r"pin\s+this", r"save\s+this",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QualityDimension(str, Enum):
    """The seven dimensions of content quality scoring."""

    READABILITY = "readability"
    EEAT = "eeat"
    SEO = "seo"
    STRUCTURE = "structure"
    ORIGINALITY = "originality"
    ENGAGEMENT = "engagement"
    VOICE_MATCH = "voice_match"


class QualityGrade(str, Enum):
    """Letter grade mapped from 0-10 numeric score."""

    A_PLUS = "A+"   # 9.0-10.0
    A = "A"         # 8.0-8.9
    B = "B"         # 7.0-7.9
    C = "C"         # 6.0-6.9
    D = "D"         # 5.0-5.9
    F = "F"         # 0-4.9


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: QualityDimension
    score: float           # 0-10
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 2),
            "details": self.details,
            "suggestions": self.suggestions,
        }


@dataclass
class QualityReport:
    """Complete quality assessment for an article."""

    article_id: str
    site_id: str
    title: str
    overall_score: float        # 0-10, weighted average
    grade: QualityGrade
    passed: bool                # overall >= threshold
    dimensions: Dict[QualityDimension, DimensionScore] = field(default_factory=dict)
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    reading_level: str = "medium"   # easy, medium, hard, expert
    estimated_reading_time_minutes: float = 0.0
    duplicate_score: float = 0.0    # 0-1, how similar to existing content
    top_suggestions: List[str] = field(default_factory=list)
    scored_at: str = ""             # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "site_id": self.site_id,
            "title": self.title,
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade.value,
            "passed": self.passed,
            "dimensions": {
                dim.value: ds.to_dict()
                for dim, ds in self.dimensions.items()
            },
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "reading_level": self.reading_level,
            "estimated_reading_time_minutes": round(
                self.estimated_reading_time_minutes, 1
            ),
            "duplicate_score": round(self.duplicate_score, 3),
            "top_suggestions": self.top_suggestions,
            "scored_at": self.scored_at,
        }

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            f"{'=' * 70}",
            f"  CONTENT QUALITY REPORT",
            f"{'=' * 70}",
            f"",
            f"  Title:          {self.title[:55]}",
            f"  Site:           {self.site_id}",
            f"  Article ID:     {self.article_id[:12]}",
            f"  Scored At:      {self.scored_at}",
            f"",
            f"  Overall Score:  {self.overall_score:.1f}/10  ({self.grade.value})",
            f"  Gate:           {'PASSED' if self.passed else 'FAILED'}",
            f"",
            f"  Word Count:     {self.word_count:,}",
            f"  Sentences:      {self.sentence_count:,}",
            f"  Paragraphs:     {self.paragraph_count:,}",
            f"  Avg Sent Len:   {self.avg_sentence_length:.1f} words",
            f"  Reading Level:  {self.reading_level}",
            f"  Reading Time:   {self.estimated_reading_time_minutes:.1f} min",
            f"  Duplicate:      {self.duplicate_score:.1%}",
            f"",
            f"  Dimension Scores:",
        ]

        for dim in QualityDimension:
            ds = self.dimensions.get(dim)
            if ds:
                label = dim.value.replace("_", " ").title()
                bar = "#" * int(ds.score) + "-" * (10 - int(ds.score))
                lines.append(f"    {label:<15} {ds.score:>5.1f}/10  [{bar}]")

        if self.top_suggestions:
            lines.append(f"")
            lines.append(f"  Top Suggestions:")
            for i, s in enumerate(self.top_suggestions[:5], 1):
                lines.append(f"    {i}. {s}")

        lines.append(f"")
        lines.append(f"{'=' * 70}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time in ISO 8601."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_json(path: Path, data: Any) -> None:
    """Write JSON atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from path, returning default if missing or corrupt."""
    if not path.exists():
        return default if default is not None else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return default if default is not None else []


def _run_sync(coro):
    """Run an async coroutine synchronously.

    Handles the case where we are already inside an event loop (e.g.,
    Jupyter notebook, nested async call).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _score_to_grade(score: float) -> QualityGrade:
    """Convert a 0-10 numeric score to a letter grade."""
    if score >= 9.0:
        return QualityGrade.A_PLUS
    elif score >= 8.0:
        return QualityGrade.A
    elif score >= 7.0:
        return QualityGrade.B
    elif score >= 6.0:
        return QualityGrade.C
    elif score >= 5.0:
        return QualityGrade.D
    else:
        return QualityGrade.F


def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# HTML Stripping
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Strip HTML tags and extract plain text, preserving block boundaries."""

    def __init__(self):
        super().__init__()
        self._pieces: List[str] = []
        self._headings: List[Tuple[str, str]] = []  # (tag, text)
        self._current_heading_tag: Optional[str] = None
        self._current_heading_text: List[str] = []
        self._in_heading = False
        self._block_tags = {
            "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "blockquote", "tr", "br", "hr",
        }
        self._list_tags = {"ul", "ol"}
        self._list_count = 0
        self._img_count = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._block_tags:
            self._pieces.append("\n")
        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = True
            self._current_heading_tag = tag_lower
            self._current_heading_text = []
        if tag_lower in self._list_tags:
            self._list_count += 1
        if tag_lower == "img":
            self._img_count += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._block_tags:
            self._pieces.append("\n")
        if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if self._in_heading:
                heading_text = "".join(self._current_heading_text).strip()
                if heading_text:
                    self._headings.append(
                        (self._current_heading_tag or tag_lower, heading_text)
                    )
                self._in_heading = False
                self._current_heading_tag = None

    def handle_data(self, data: str) -> None:
        self._pieces.append(data)
        if self._in_heading:
            self._current_heading_text.append(data)

    def get_text(self) -> str:
        return "".join(self._pieces)

    def get_headings(self) -> List[Tuple[str, str]]:
        return self._headings

    def get_list_count(self) -> int:
        return self._list_count

    def get_img_count(self) -> int:
        return self._img_count


def _strip_html(html: str) -> Tuple[str, List[Tuple[str, str]], int, int]:
    """Strip HTML and return (text, headings, list_count, img_count)."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    text = stripper.get_text()
    # Normalize whitespace within lines but preserve paragraph breaks
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = " ".join(line.split())
        if stripped:
            cleaned.append(stripped)
    return (
        "\n".join(cleaned),
        stripper.get_headings(),
        stripper.get_list_count(),
        stripper.get_img_count(),
    )


# ---------------------------------------------------------------------------
# Text Analysis Utilities (Pure Python, no NLTK)
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    """Count syllables in a word using vowel-group heuristic.

    Rules:
    1. Count groups of consecutive vowels (a, e, i, o, u, y)
    2. Subtract 1 for silent-e at end (if word > 2 chars)
    3. Add 1 for -le ending preceded by consonant
    4. Minimum 1 syllable per word
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Remove non-alpha characters
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    vowels = set("aeiouy")
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Silent e at end
    if word.endswith("e") and len(word) > 2 and word[-2] not in vowels:
        count -= 1

    # -le ending preceded by consonant (e.g., "table", "little")
    if (
        len(word) > 2
        and word.endswith("le")
        and word[-3] not in vowels
        and count == 0
    ):
        count += 1

    # -ed endings that are silent (walked, played but not wanted, needed)
    if (
        word.endswith("ed")
        and len(word) > 3
        and word[-3] not in "dt"
        and count > 1
    ):
        count -= 1

    return max(1, count)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex heuristics.

    Handles abbreviations (Mr., Dr., etc.), ellipses, and numbers.
    """
    # Protect common abbreviations
    protected = text
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
        "vs.", "etc.", "i.e.", "e.g.", "a.m.", "p.m.", "U.S.",
        "U.K.", "Fig.", "No.", "Vol.",
    ]
    placeholders: Dict[str, str] = {}
    for i, abbr in enumerate(abbreviations):
        ph = f"__ABBR{i}__"
        placeholders[ph] = abbr
        protected = protected.replace(abbr, ph)

    # Split on sentence-ending punctuation followed by whitespace or end
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)

    # Restore abbreviations
    sentences = []
    for s in raw_sentences:
        for ph, abbr in placeholders.items():
            s = s.replace(ph, abbr)
        s = s.strip()
        if s:
            sentences.append(s)

    return sentences


def _split_words(text: str) -> List[str]:
    """Split text into words, stripping punctuation."""
    return [w for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 0]


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (non-empty lines separated by blank lines)."""
    paragraphs = []
    current = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _flesch_kincaid_grade(
    total_words: int, total_sentences: int, total_syllables: int
) -> float:
    """Calculate the Flesch-Kincaid Grade Level.

    FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    """
    if total_sentences == 0 or total_words == 0:
        return 0.0

    asl = total_words / total_sentences  # Average Sentence Length
    asw = total_syllables / total_words  # Average Syllables per Word

    grade = 0.39 * asl + 11.8 * asw - 15.59
    return max(0.0, grade)


def _flesch_reading_ease(
    total_words: int, total_sentences: int, total_syllables: int
) -> float:
    """Calculate the Flesch Reading Ease score (0-100, higher = easier).

    FRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    """
    if total_sentences == 0 or total_words == 0:
        return 0.0

    asl = total_words / total_sentences
    asw = total_syllables / total_words

    ease = 206.835 - 1.015 * asl - 84.6 * asw
    return max(0.0, min(100.0, ease))


def _reading_level_from_grade(fk_grade: float) -> str:
    """Map Flesch-Kincaid grade level to a reading level label."""
    if fk_grade <= 6.0:
        return "easy"
    elif fk_grade <= 10.0:
        return "medium"
    elif fk_grade <= 14.0:
        return "hard"
    else:
        return "expert"


def _detect_passive_voice(sentences: List[str]) -> Tuple[int, List[str]]:
    """Detect passive voice constructions in sentences.

    Returns (count, list_of_passive_sentences).
    Uses pattern: auxiliary + (adverb)? + past_participle.
    """
    # Past participle pattern: word ending in -ed, -en, -t (common patterns)
    pp_pattern = re.compile(
        r"\b(is|are|was|were|be|been|being|get|gets|got|gotten)\b"
        r"\s+(?:\w+\s+)?"          # optional adverb
        r"(\w+(?:ed|en|t|wn|ng))\b",  # past participle
        re.IGNORECASE,
    )
    passive_count = 0
    passive_sentences = []

    for sentence in sentences:
        if pp_pattern.search(sentence):
            passive_count += 1
            passive_sentences.append(sentence[:80])

    return passive_count, passive_sentences


def _count_signal_matches(
    text_lower: str, signals: List[str]
) -> Tuple[int, List[str]]:
    """Count how many signal phrases appear in text. Returns (count, matched)."""
    matched = []
    for signal in signals:
        if signal in text_lower:
            matched.append(signal)
    return len(matched), matched


# ---------------------------------------------------------------------------
# Shingle-Based Fingerprinting for Duplicate Detection
# ---------------------------------------------------------------------------


def _build_shingles(text: str, size: int = SHINGLE_SIZE) -> Set[str]:
    """Build a set of word-level shingles from text.

    A shingle is a contiguous sequence of `size` words. Used for
    Jaccard similarity comparison.
    """
    words = _split_words(text.lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()

    shingles = set()
    for i in range(len(words) - size + 1):
        shingle = " ".join(words[i : i + size])
        shingles.add(shingle)

    return shingles


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets (0-1)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _fingerprint_to_list(shingles: Set[str]) -> List[str]:
    """Convert shingle set to a sorted list for JSON serialization.

    Stores hashes instead of raw shingles to save space.
    """
    return sorted(
        hashlib.md5(s.encode("utf-8")).hexdigest()[:12] for s in shingles
    )


def _list_to_fingerprint(hashes: List[str]) -> Set[str]:
    """Convert stored hash list back to a set for comparison."""
    return set(hashes)


# ---------------------------------------------------------------------------
# ContentQualityScorer
# ---------------------------------------------------------------------------


class ContentQualityScorer:
    """Scores article content across 7 quality dimensions.

    All scoring is pure Python -- no external NLP libraries required.

    Attributes:
        _history: Bounded list of scoring history (max 2000 entries).
        _threshold: Minimum passing score for quality gate (default 6.0).
        _dimension_weights: Customizable weights for each dimension.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._history: List[Dict] = _load_json(HISTORY_FILE, [])
        config = _load_json(CONFIG_FILE, {})

        self._threshold: float = threshold or config.get(
            "threshold", DEFAULT_THRESHOLD
        )

        stored_weights = config.get("weights", {})
        self._dimension_weights: Dict[QualityDimension, float] = {}
        for dim in QualityDimension:
            w = DEFAULT_WEIGHTS.get(dim.value, 1.0 / len(QualityDimension))
            if weights and dim.value in weights:
                w = weights[dim.value]
            elif stored_weights and dim.value in stored_weights:
                w = stored_weights[dim.value]
            self._dimension_weights[dim] = w

        # Normalize weights to sum to 1.0
        total_w = sum(self._dimension_weights.values())
        if total_w > 0:
            for dim in self._dimension_weights:
                self._dimension_weights[dim] /= total_w

        logger.debug(
            "ContentQualityScorer initialized: threshold=%.1f, weights=%s",
            self._threshold,
            {d.value: round(w, 3) for d, w in self._dimension_weights.items()},
        )

    # ------------------------------------------------------------------
    # Main Scoring Interface
    # ------------------------------------------------------------------

    async def score(
        self,
        content: str,
        title: str,
        site_id: str,
        keywords: Optional[List[str]] = None,
    ) -> QualityReport:
        """Score content across all quality dimensions.

        Args:
            content: Article content (HTML or plain text).
            title: Article title/headline.
            site_id: Target site identifier.
            keywords: Optional focus keywords for SEO scoring.

        Returns:
            A QualityReport with scores, grade, and suggestions.
        """
        article_id = str(uuid.uuid4())
        keywords = keywords or []

        # Strip HTML and extract structural elements
        plain_text, headings, list_count, img_count = _strip_html(content)

        # Basic text metrics
        words = _split_words(plain_text)
        sentences = _split_sentences(plain_text)
        paragraphs = _split_paragraphs(plain_text)

        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0.0
        )

        # Syllable counting for readability
        total_syllables = sum(_count_syllables(w) for w in words)

        # Flesch-Kincaid grade level
        fk_grade = _flesch_kincaid_grade(
            word_count, sentence_count, total_syllables
        )
        reading_level = _reading_level_from_grade(fk_grade)

        # Reading time
        reading_time = word_count / READING_SPEED_WPM if word_count > 0 else 0.0

        # Score each dimension
        dim_readability = self._score_readability(
            plain_text, words, sentences, paragraphs,
            total_syllables, fk_grade,
        )
        dim_eeat = self._score_eeat(plain_text, title)
        dim_seo = self._score_seo(
            plain_text, title, keywords, headings, word_count,
        )
        dim_structure = self._score_structure(
            plain_text, headings, paragraphs, list_count, img_count,
            word_count, sentences,
        )
        dim_originality = self._score_originality(plain_text, site_id)
        dim_engagement = self._score_engagement(
            plain_text, sentences, words, headings,
        )
        dim_voice = await self._score_voice_match(plain_text, site_id)

        dimensions: Dict[QualityDimension, DimensionScore] = {
            QualityDimension.READABILITY: dim_readability,
            QualityDimension.EEAT: dim_eeat,
            QualityDimension.SEO: dim_seo,
            QualityDimension.STRUCTURE: dim_structure,
            QualityDimension.ORIGINALITY: dim_originality,
            QualityDimension.ENGAGEMENT: dim_engagement,
            QualityDimension.VOICE_MATCH: dim_voice,
        }

        # Weighted overall score
        overall_score = 0.0
        for dim, ds in dimensions.items():
            weight = self._dimension_weights.get(dim, 0.0)
            overall_score += ds.score * weight

        overall_score = _clamp(overall_score)
        grade = _score_to_grade(overall_score)
        passed = overall_score >= self._threshold

        # Collect top suggestions from all dimensions
        all_suggestions: List[Tuple[float, str]] = []
        for dim, ds in dimensions.items():
            # Weight suggestions by how far below 10 the dimension scores
            urgency = 10.0 - ds.score
            for suggestion in ds.suggestions:
                all_suggestions.append((urgency, suggestion))

        all_suggestions.sort(key=lambda x: -x[0])
        top_suggestions = [s for _, s in all_suggestions[:5]]

        # Duplicate score from originality dimension
        duplicate_score = dim_originality.details.get("jaccard_max", 0.0)

        report = QualityReport(
            article_id=article_id,
            site_id=site_id,
            title=title,
            overall_score=overall_score,
            grade=grade,
            passed=passed,
            dimensions=dimensions,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            reading_level=reading_level,
            estimated_reading_time_minutes=reading_time,
            duplicate_score=duplicate_score,
            top_suggestions=top_suggestions,
            scored_at=_now_iso(),
        )

        # Persist to history (bounded)
        self._append_history(report)

        logger.info(
            "Scored '%s' for site '%s': %.1f/10 (%s) -- %s",
            title[:40], site_id, overall_score, grade.value,
            "PASSED" if passed else "FAILED",
        )

        return report

    def score_sync(
        self,
        content: str,
        title: str,
        site_id: str,
        keywords: Optional[List[str]] = None,
    ) -> QualityReport:
        """Synchronous wrapper for score()."""
        return _run_sync(
            self.score(content, title, site_id, keywords)
        )

    # ------------------------------------------------------------------
    # Dimension Scorers
    # ------------------------------------------------------------------

    def _score_readability(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        paragraphs: List[str],
        total_syllables: int,
        fk_grade: float,
    ) -> DimensionScore:
        """Score readability using Flesch-Kincaid and structural analysis.

        Optimal range for web content: FK grade 6-10 (middle school to
        early high school). Penalizes both overly simple (grade < 4)
        and overly complex (grade > 14) content.
        """
        score = 10.0
        details: Dict[str, Any] = {}
        suggestions: List[str] = []

        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0.0
        )

        # Flesch Reading Ease
        fre = _flesch_reading_ease(word_count, sentence_count, total_syllables)
        details["flesch_reading_ease"] = round(fre, 1)
        details["flesch_kincaid_grade"] = round(fk_grade, 1)
        details["avg_sentence_length"] = round(avg_sentence_length, 1)

        # FK grade scoring: sweet spot is 6-10
        if fk_grade < 4.0:
            penalty = (4.0 - fk_grade) * 0.5
            score -= penalty
            suggestions.append(
                f"Content reads too simply (grade {fk_grade:.1f}). "
                f"Add more varied vocabulary and complex sentence structures."
            )
        elif fk_grade <= 6.0:
            # Slightly below sweet spot but fine
            pass
        elif fk_grade <= 10.0:
            # Perfect range -- no penalty
            pass
        elif fk_grade <= 14.0:
            penalty = (fk_grade - 10.0) * 0.5
            score -= penalty
            suggestions.append(
                f"Content is moderately complex (grade {fk_grade:.1f}). "
                f"Simplify some sentences for broader readability."
            )
        else:
            penalty = 2.0 + (fk_grade - 14.0) * 0.3
            score -= penalty
            suggestions.append(
                f"Content is very complex (grade {fk_grade:.1f}). "
                f"Break long sentences, use simpler words where possible."
            )

        # Average sentence length: ideal 15-20 words
        if avg_sentence_length > 25:
            penalty = (avg_sentence_length - 25) * 0.15
            score -= min(penalty, 2.0)
            suggestions.append(
                f"Average sentence length is {avg_sentence_length:.0f} words. "
                f"Aim for 15-20 words per sentence."
            )
        elif avg_sentence_length < 8 and sentence_count > 3:
            penalty = (8 - avg_sentence_length) * 0.2
            score -= min(penalty, 1.5)
            suggestions.append(
                f"Sentences are very short (avg {avg_sentence_length:.0f} words). "
                f"Vary sentence length for better flow."
            )

        # Sentence length variance (check for monotony)
        if sentence_count > 5:
            sent_lengths = [len(_split_words(s)) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(
                sent_lengths
            )
            std_dev = math.sqrt(variance)
            details["sentence_length_std_dev"] = round(std_dev, 1)

            if std_dev < 3.0:
                score -= 0.5
                suggestions.append(
                    "Sentence lengths are very uniform. Vary between short "
                    "punchy sentences and longer explanatory ones."
                )

        # Passive voice detection
        passive_count, passive_examples = _detect_passive_voice(sentences)
        passive_ratio = (
            passive_count / sentence_count if sentence_count > 0 else 0.0
        )
        details["passive_voice_count"] = passive_count
        details["passive_voice_ratio"] = round(passive_ratio, 3)

        if passive_ratio > 0.30:
            penalty = (passive_ratio - 0.30) * 8.0
            score -= min(penalty, 2.0)
            suggestions.append(
                f"{passive_count} of {sentence_count} sentences use passive voice "
                f"({passive_ratio:.0%}). Aim for under 20%."
            )
        elif passive_ratio > 0.20:
            score -= 0.5
            suggestions.append(
                f"Passive voice detected in {passive_ratio:.0%} of sentences. "
                f"Consider converting some to active voice."
            )

        # Paragraph length distribution
        if paragraphs:
            para_lengths = [len(_split_words(p)) for p in paragraphs]
            long_paras = sum(1 for l in para_lengths if l > 150)
            details["long_paragraphs"] = long_paras

            if long_paras > 0:
                penalty = long_paras * 0.5
                score -= min(penalty, 1.5)
                suggestions.append(
                    f"{long_paras} paragraph(s) exceed 150 words. "
                    f"Break them into smaller chunks for web readability."
                )

        # Big words ratio (3+ syllables)
        if word_count > 0:
            big_words = sum(
                1 for w in words if _count_syllables(w) >= 3
            )
            big_ratio = big_words / word_count
            details["big_words_ratio"] = round(big_ratio, 3)

            if big_ratio > 0.30:
                score -= 1.0
                suggestions.append(
                    f"{big_ratio:.0%} of words have 3+ syllables. "
                    f"Replace some with simpler alternatives."
                )

        details["word_count"] = word_count
        details["sentence_count"] = sentence_count
        details["paragraph_count"] = len(paragraphs)

        return DimensionScore(
            dimension=QualityDimension.READABILITY,
            score=_clamp(score),
            details=details,
            suggestions=suggestions,
        )

    def _score_eeat(self, text: str, title: str) -> DimensionScore:
        """Score E-E-A-T signals in content.

        Checks for:
        - Experience: First-person anecdotes, personal testing
        - Expertise: Data, statistics, citations, research references
        - Authoritativeness: Credentials, qualifications
        - Trustworthiness: Disclaimers, sources, transparency
        """
        score = 5.0  # Start at mid-point; signals push up, absence pulls down
        details: Dict[str, Any] = {}
        suggestions: List[str] = []
        text_lower = text.lower()

        # Experience signals
        exp_count, exp_matched = _count_signal_matches(
            text_lower, EXPERIENCE_SIGNALS
        )
        details["experience_signals"] = exp_count
        details["experience_matched"] = exp_matched[:5]

        if exp_count >= 5:
            score += 1.5
        elif exp_count >= 3:
            score += 1.0
        elif exp_count >= 1:
            score += 0.5
        else:
            score -= 0.5
            suggestions.append(
                "Add first-person experience markers (e.g., 'I found', "
                "'in my experience', 'I tested') to boost E-E-A-T."
            )

        # Expertise signals
        exp_count2, exp_matched2 = _count_signal_matches(
            text_lower, EXPERTISE_SIGNALS
        )
        details["expertise_signals"] = exp_count2
        details["expertise_matched"] = exp_matched2[:5]

        if exp_count2 >= 5:
            score += 1.5
        elif exp_count2 >= 3:
            score += 1.0
        elif exp_count2 >= 1:
            score += 0.5
        else:
            score -= 0.5
            suggestions.append(
                "Include data, statistics, or research references to "
                "demonstrate expertise."
            )

        # Authority signals
        auth_count, auth_matched = _count_signal_matches(
            text_lower, AUTHORITY_SIGNALS
        )
        details["authority_signals"] = auth_count
        details["authority_matched"] = auth_matched[:5]

        if auth_count >= 3:
            score += 1.0
        elif auth_count >= 1:
            score += 0.5

        # Trust signals
        trust_count, trust_matched = _count_signal_matches(
            text_lower, TRUST_SIGNALS
        )
        details["trust_signals"] = trust_count
        details["trust_matched"] = trust_matched[:5]

        if trust_count >= 3:
            score += 1.5
        elif trust_count >= 2:
            score += 1.0
        elif trust_count >= 1:
            score += 0.5
        else:
            suggestions.append(
                "Add trust markers: disclaimers, source citations, or "
                "disclosure statements."
            )

        # Check for specific numbers/data in content
        number_pattern = re.compile(r"\b\d+(?:\.\d+)?%|\b\d{2,}\b")
        numbers_found = len(number_pattern.findall(text))
        details["numbers_and_data_points"] = numbers_found

        if numbers_found >= 5:
            score += 0.5
        elif numbers_found == 0:
            score -= 0.5
            suggestions.append(
                "Include specific numbers, percentages, or data points "
                "to strengthen credibility."
            )

        # External source indicators
        source_patterns = re.compile(
            r"(?:https?://|www\.|source:|according to \w+)",
            re.IGNORECASE,
        )
        source_count = len(source_patterns.findall(text))
        details["source_references"] = source_count

        if source_count >= 3:
            score += 0.5
        elif source_count == 0:
            suggestions.append(
                "Reference external sources or include links to "
                "build authoritativeness."
            )

        return DimensionScore(
            dimension=QualityDimension.EEAT,
            score=_clamp(score),
            details=details,
            suggestions=suggestions,
        )

    def _score_seo(
        self,
        text: str,
        title: str,
        keywords: List[str],
        headings: List[Tuple[str, str]],
        word_count: int,
    ) -> DimensionScore:
        """Score SEO optimization.

        Checks:
        - Keyword in title and first paragraph
        - H2/H3 heading structure for featured snippets
        - Content length adequacy
        - FAQ section presence
        - Meta description length estimate (first paragraph)
        - Internal link opportunity mentions
        """
        score = 7.0  # Start at decent baseline
        details: Dict[str, Any] = {}
        suggestions: List[str] = []
        text_lower = text.lower()
        title_lower = title.lower()

        # --- Keyword analysis ---
        if keywords:
            primary_keyword = keywords[0].lower()
            details["primary_keyword"] = keywords[0]

            # Keyword in title
            kw_in_title = primary_keyword in title_lower
            details["keyword_in_title"] = kw_in_title
            if kw_in_title:
                score += 0.5
            else:
                score -= 1.0
                suggestions.append(
                    f"Include focus keyword '{keywords[0]}' in the title."
                )

            # Keyword in first paragraph (first 100 words)
            first_para_words = text_lower.split()[:100]
            first_para_text = " ".join(first_para_words)
            kw_in_first_para = primary_keyword in first_para_text
            details["keyword_in_first_paragraph"] = kw_in_first_para
            if kw_in_first_para:
                score += 0.5
            else:
                score -= 1.0
                suggestions.append(
                    f"Include focus keyword '{keywords[0]}' in the first "
                    f"paragraph (first 100 words)."
                )

            # Keyword density (ideal: 0.5% - 2.5%)
            if word_count > 0:
                kw_words = primary_keyword.split()
                kw_occurrences = text_lower.count(primary_keyword)
                density = (kw_occurrences * len(kw_words)) / word_count
                details["keyword_density"] = round(density * 100, 2)

                if density < 0.003:
                    score -= 0.5
                    suggestions.append(
                        f"Keyword density is very low ({density:.1%}). "
                        f"Use the focus keyword a few more times."
                    )
                elif density > 0.03:
                    score -= 0.5
                    suggestions.append(
                        f"Keyword density is high ({density:.1%}). "
                        f"Reduce keyword usage to avoid over-optimization."
                    )

            # Keyword in headings
            kw_in_headings = any(
                primary_keyword in h_text.lower()
                for _, h_text in headings
            )
            details["keyword_in_headings"] = kw_in_headings
            if kw_in_headings:
                score += 0.3
            else:
                suggestions.append(
                    f"Include focus keyword '{keywords[0]}' in at least "
                    f"one H2 or H3 heading."
                )

            # Secondary keywords
            if len(keywords) > 1:
                secondary_found = sum(
                    1 for kw in keywords[1:]
                    if kw.lower() in text_lower
                )
                details["secondary_keywords_found"] = secondary_found
                details["secondary_keywords_total"] = len(keywords) - 1
        else:
            score -= 1.0
            suggestions.append(
                "No focus keywords provided. Define primary and secondary "
                "keywords for SEO optimization."
            )

        # --- Heading structure ---
        h2_count = sum(1 for tag, _ in headings if tag == "h2")
        h3_count = sum(1 for tag, _ in headings if tag == "h3")
        details["h2_count"] = h2_count
        details["h3_count"] = h3_count

        if h2_count == 0:
            score -= 1.5
            suggestions.append(
                "No H2 headings found. Add H2 subheadings to structure "
                "content for SEO and featured snippets."
            )
        elif h2_count < 3 and word_count > 800:
            score -= 0.5
            suggestions.append(
                f"Only {h2_count} H2 heading(s) for {word_count} words. "
                f"Add more H2 sections for better structure."
            )

        # --- Content length ---
        details["word_count"] = word_count

        if word_count < 300:
            score -= 2.0
            suggestions.append(
                f"Content is thin ({word_count} words). "
                f"Aim for at least 1,000 words for SEO value."
            )
        elif word_count < 800:
            score -= 1.0
            suggestions.append(
                f"Content is short ({word_count} words). "
                f"Consider expanding to 1,500+ words for better rankings."
            )
        elif word_count >= 1500:
            score += 0.5

        # --- FAQ section ---
        has_faq = any(
            "faq" in h_text.lower()
            or "frequently asked" in h_text.lower()
            or "common questions" in h_text.lower()
            for _, h_text in headings
        )
        details["has_faq_section"] = has_faq

        if has_faq:
            score += 0.5
        else:
            suggestions.append(
                "Add a FAQ section with schema markup for rich snippet "
                "opportunities."
            )

        # --- Meta description estimate (first paragraph length) ---
        paragraphs = _split_paragraphs(text)
        if paragraphs:
            first_para_len = len(paragraphs[0])
            details["first_paragraph_chars"] = first_para_len

            if first_para_len < 50:
                suggestions.append(
                    "First paragraph is very short. Write a compelling "
                    "opening of 120-160 characters for meta description."
                )
            elif first_para_len > 300:
                suggestions.append(
                    "First paragraph is long. Keep it under 160 characters "
                    "so it works as a meta description."
                )

        return DimensionScore(
            dimension=QualityDimension.SEO,
            score=_clamp(score),
            details=details,
            suggestions=suggestions,
        )

    def _score_structure(
        self,
        text: str,
        headings: List[Tuple[str, str]],
        paragraphs: List[str],
        list_count: int,
        img_count: int,
        word_count: int,
        sentences: List[str],
    ) -> DimensionScore:
        """Score content structure quality.

        Checks:
        - H2/H3 hierarchy (no skipping levels)
        - Paragraph length distribution (not too long, not too short)
        - List usage (bullet/numbered lists for scannability)
        - Image placeholders
        - Introduction and conclusion presence
        - Table of contents readiness
        """
        score = 7.0
        details: Dict[str, Any] = {}
        suggestions: List[str] = []

        # --- Heading hierarchy ---
        heading_levels = [tag for tag, _ in headings]
        details["heading_count"] = len(headings)
        details["heading_levels"] = heading_levels

        if headings:
            # Check for proper hierarchy (no skipping from h2 to h4)
            hierarchy_ok = True
            prev_level = 1  # Assume h1 is the title
            for tag in heading_levels:
                level = int(tag[1])
                if level > prev_level + 1:
                    hierarchy_ok = False
                    break
                prev_level = level

            details["heading_hierarchy_valid"] = hierarchy_ok
            if not hierarchy_ok:
                score -= 1.0
                suggestions.append(
                    "Heading hierarchy skips levels (e.g., H2 to H4). "
                    "Use sequential heading levels for accessibility and SEO."
                )
        else:
            if word_count > 300:
                score -= 2.0
                suggestions.append(
                    "No headings found. Add H2/H3 headings to break up "
                    "content and improve scannability."
                )

        # --- Paragraph analysis ---
        if paragraphs:
            para_word_counts = [len(_split_words(p)) for p in paragraphs]
            avg_para_len = (
                sum(para_word_counts) / len(para_word_counts)
                if para_word_counts else 0
            )
            details["avg_paragraph_length"] = round(avg_para_len, 1)
            details["paragraph_count"] = len(paragraphs)

            # Check for wall-of-text paragraphs
            wall_count = sum(1 for l in para_word_counts if l > 100)
            details["wall_of_text_paragraphs"] = wall_count

            if wall_count > 0:
                penalty = wall_count * 0.5
                score -= min(penalty, 2.0)
                suggestions.append(
                    f"{wall_count} paragraph(s) exceed 100 words. "
                    f"Break into smaller paragraphs (3-5 sentences each)."
                )

            # Very short paragraphs (possibly incomplete)
            tiny_count = sum(1 for l in para_word_counts if l < 5)
            if tiny_count > len(paragraphs) * 0.3 and len(paragraphs) > 3:
                score -= 0.5
                suggestions.append(
                    "Many very short paragraphs (under 5 words). "
                    "Combine related short fragments."
                )

            # Ideal paragraph length variance
            if len(para_word_counts) > 3:
                mean_p = sum(para_word_counts) / len(para_word_counts)
                var_p = sum(
                    (l - mean_p) ** 2 for l in para_word_counts
                ) / len(para_word_counts)
                std_p = math.sqrt(var_p)
                details["paragraph_length_std_dev"] = round(std_p, 1)
        else:
            score -= 2.0
            suggestions.append(
                "No paragraphs detected. Structure content into clear "
                "paragraphs."
            )

        # --- Lists ---
        details["list_count"] = list_count

        if list_count == 0 and word_count > 500:
            score -= 0.5
            suggestions.append(
                "No bullet or numbered lists found. Add lists to improve "
                "scannability and featured snippet potential."
            )
        elif list_count >= 2:
            score += 0.5

        # --- Images ---
        details["img_count"] = img_count

        if img_count == 0 and word_count > 500:
            score -= 0.5
            suggestions.append(
                "No images detected. Add relevant images with alt text "
                "to enhance engagement and SEO."
            )
        elif img_count >= 3:
            score += 0.3

        # --- Introduction ---
        has_intro = False
        if paragraphs and len(paragraphs) > 1:
            # Check if content starts with a paragraph before the first heading
            first_heading_pos = text.find(
                headings[0][1] if headings else "\x00"
            )
            first_para_text = paragraphs[0]
            if first_heading_pos > len(first_para_text) or not headings:
                has_intro = True
                if len(_split_words(first_para_text)) >= 20:
                    score += 0.3
            elif len(_split_words(first_para_text)) >= 20:
                has_intro = True

        details["has_introduction"] = has_intro
        if not has_intro:
            suggestions.append(
                "Add an introduction paragraph before the first heading "
                "to set context and include the focus keyword."
            )

        # --- Conclusion ---
        has_conclusion = False
        if headings:
            last_heading_text = headings[-1][1].lower()
            conclusion_keywords = [
                "conclusion", "summary", "final thoughts",
                "wrapping up", "takeaway", "key takeaways",
                "bottom line", "in closing",
            ]
            has_conclusion = any(
                kw in last_heading_text for kw in conclusion_keywords
            )
        details["has_conclusion"] = has_conclusion

        if not has_conclusion and word_count > 800:
            score -= 0.3
            suggestions.append(
                "Add a conclusion section (e.g., 'Key Takeaways' or "
                "'Final Thoughts') to summarize the article."
            )

        # --- Table of Contents readiness ---
        toc_ready = len(headings) >= 4
        details["toc_ready"] = toc_ready
        if toc_ready:
            score += 0.2

        return DimensionScore(
            dimension=QualityDimension.STRUCTURE,
            score=_clamp(score),
            details=details,
            suggestions=suggestions,
        )

    def _score_originality(
        self, text: str, site_id: str
    ) -> DimensionScore:
        """Score content originality via shingle-based fingerprinting.

        Compares article shingles against the corpus of previously scored
        articles for the same site. High Jaccard similarity indicates
        duplicate or near-duplicate content.
        """
        score = 10.0
        details: Dict[str, Any] = {}
        suggestions: List[str] = []
        text_lower = text.lower()

        # Build shingles for this article
        shingles = _build_shingles(text)
        details["shingle_count"] = len(shingles)

        # Check for flagged phrases
        flagged_found = []
        for phrase in FLAGGED_PHRASES:
            if phrase in text_lower:
                flagged_found.append(phrase)

        details["flagged_phrases_count"] = len(flagged_found)
        details["flagged_phrases"] = flagged_found[:10]

        if len(flagged_found) >= 8:
            score -= 3.0
            suggestions.append(
                f"Content contains {len(flagged_found)} cliche/AI-flagged "
                f"phrases. Rewrite to sound more natural and original."
            )
        elif len(flagged_found) >= 5:
            score -= 2.0
            suggestions.append(
                f"Found {len(flagged_found)} overused phrases "
                f"({', '.join(flagged_found[:3])}...). Replace with "
                f"original language."
            )
        elif len(flagged_found) >= 3:
            score -= 1.0
            suggestions.append(
                f"Found {len(flagged_found)} common phrases that may "
                f"signal AI-generated content. Consider alternatives."
            )
        elif len(flagged_found) >= 1:
            score -= 0.3

        # Check duplicates against corpus
        jaccard_max, closest_id = self._check_duplicates(shingles, site_id)
        details["jaccard_max"] = round(jaccard_max, 4)
        details["closest_article_id"] = closest_id

        if jaccard_max > 0.70:
            score -= 4.0
            suggestions.append(
                f"Content is {jaccard_max:.0%} similar to an existing "
                f"article (ID: {closest_id[:8] if closest_id else '?'}). "
                f"This may be a near-duplicate."
            )
        elif jaccard_max > 0.50:
            score -= 2.5
            suggestions.append(
                f"Content shares {jaccard_max:.0%} overlap with existing "
                f"content. Ensure this is sufficiently differentiated."
            )
        elif jaccard_max > 0.30:
            score -= 1.0
            suggestions.append(
                f"Moderate similarity ({jaccard_max:.0%}) with existing "
                f"content. Add unique insights or angles."
            )
        elif jaccard_max > 0.15:
            score -= 0.3

        # Vocabulary richness (unique words / total words)
        words = _split_words(text.lower())
        if len(words) > 50:
            unique_ratio = len(set(words)) / len(words)
            details["vocabulary_richness"] = round(unique_ratio, 3)

            if unique_ratio < 0.30:
                score -= 1.0
                suggestions.append(
                    f"Vocabulary diversity is low ({unique_ratio:.0%} unique "
                    f"words). Use more varied language."
                )
            elif unique_ratio < 0.40:
                score -= 0.5
                suggestions.append(
                    "Consider using more diverse vocabulary to improve "
                    "originality signals."
                )
            elif unique_ratio > 0.60:
                score += 0.3

        # Repetitive phrase detection (any 3-word sequence appearing 5+ times)
        if len(words) > 100:
            trigrams = Counter()
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                trigrams[trigram] += 1

            repetitive = [
                (phrase, count)
                for phrase, count in trigrams.most_common(10)
                if count >= 5
            ]
            details["repetitive_trigrams"] = len(repetitive)

            if repetitive:
                score -= min(len(repetitive) * 0.3, 1.5)
                top_rep = repetitive[0]
                suggestions.append(
                    f"Phrase '{top_rep[0]}' repeats {top_rep[1]} times. "
                    f"Vary your language to avoid redundancy."
                )

        return DimensionScore(
            dimension=QualityDimension.ORIGINALITY,
            score=_clamp(score),
            details=details,
            suggestions=suggestions,
        )

    def _score_engagement(
        self,
        text: str,
        sentences: List[str],
        words: List[str],
        headings: List[Tuple[str, str]],
    ) -> DimensionScore:
        """Score content engagement potential.

        Checks:
        - Hook strength (first 2 sentences)
        - CTA presence
        - Question usage throughout article
        - Story/narrative elements
        - Emotional language
        """
        score = 6.0
        details: Dict[str, Any] = {}
        suggestions: List[str] = []
        text_lower = text.lower()

        # --- Hook strength (first 2 sentences) ---
        hook_score = 0.0
        if len(sentences) >= 2:
            hook_text = " ".join(sentences[:2]).lower()
            hook_words = _split_words(hook_text)

            # Check for question hook
            if "?" in sentences[0]:
                hook_score += 2.0
                details["hook_type"] = "question"
            # Check for statistic/number hook
            elif re.search(r"\d+%|\d{3,}", sentences[0]):
                hook_score += 1.5
                details["hook_type"] = "statistic"
            # Check for story hook ("when I", "imagine", "picture this")
            elif any(
                trigger in hook_text
                for trigger in [
                    "when i", "imagine", "picture this", "last year",
                    "one day", "i remember", "have you ever",
                ]
            ):
                hook_score += 2.0
                details["hook_type"] = "story"
            # Check for bold statement
            elif len(hook_words) < 15 and any(
                w in hook_text
                for w in ["never", "always", "must", "essential", "secret"]
            ):
                hook_score += 1.0
                details["hook_type"] = "bold_statement"
            else:
                details["hook_type"] = "generic"
                suggestions.append(
                    "Strengthen the opening hook. Start with a question, "
                    "surprising statistic, or a short story."
                )

            details["hook_score"] = round(hook_score, 1)
            score += min(hook_score, 2.0)
        else:
            suggestions.append(
                "Content is too short for effective hook analysis. "
                "Write at least 2-3 opening sentences."
            )

        # --- Questions throughout ---
        question_count = sum(1 for s in sentences if "?" in s)
        details["question_count"] = question_count

        if len(sentences) > 0:
            question_ratio = question_count / len(sentences)
            details["question_ratio"] = round(question_ratio, 3)

            if question_count == 0:
                score -= 0.5
                suggestions.append(
                    "No questions found in the content. Add rhetorical "
                    "questions to engage readers and guide their thinking."
                )
            elif question_count >= 3:
                score += 0.5
            elif question_ratio > 0.3:
                score -= 0.3
                suggestions.append(
                    "Too many questions may reduce authority. Balance "
                    "questions with confident statements."
                )

        # --- CTA presence ---
        cta_found = []
        for pattern in CTA_PATTERNS:
            if re.search(pattern, text_lower):
                cta_found.append(pattern.replace(r"\s+", " "))

        details["cta_count"] = len(cta_found)
        details["cta_patterns"] = cta_found[:5]

        if len(cta_found) == 0:
            score -= 1.0
            suggestions.append(
                "No call-to-action found. Add CTAs like 'subscribe', "
                "'share this', or 'leave a comment'."
            )
        elif len(cta_found) >= 2:
            score += 0.5
        elif len(cta_found) >= 1:
            score += 0.3

        # --- Story/narrative elements ---
        story_markers = [
            "once upon", "story", "anecdote", "experience",
            "journey", "when i was", "i remember", "years ago",
            "one time", "true story", "real-life", "case study",
        ]
        story_count, story_matched = _count_signal_matches(
            text_lower, story_markers
        )
        details["story_elements"] = story_count

        if story_count >= 3:
            score += 1.0
        elif story_count >= 1:
            score += 0.5
        else:
            suggestions.append(
                "Add personal anecdotes or real-life examples to "
                "make content more relatable and engaging."
            )

        # --- Emotional language ---
        emotion_words = [
            "amazing", "incredible", "powerful", "beautiful",
            "terrifying", "exciting", "heartbreaking", "inspiring",
            "shocking", "surprising", "fascinating", "love",
            "hate", "fear", "joy", "passion", "struggle",
            "triumph", "overcome", "transform", "breakthrough",
        ]
        emotion_count, _ = _count_signal_matches(text_lower, emotion_words)
        details["emotional_words"] = emotion_count

        word_count = len(words)
        if word_count > 0:
            emotion_density = emotion_count / word_count
            if emotion_density > 0.02:
                score -= 0.3
                suggestions.append(
                    "Emotional language is dense. Tone down superlatives "
                    "for more credible writing."
                )
            elif emotion_count >= 3:
                score += 0.3
            elif emotion_count == 0 and word_count > 300:
                suggestions.append(
                    "Content lacks emotional language. Add vivid, "
                    "descriptive words to connect with readers."
                )

        # --- Heading engagement ---
        if headings:
            engaging_heading_count = 0
            for _, h_text in headings:
                h_lower = h_text.lower()
                if (
                    "?" in h_text
                    or "how to" in h_lower
                    or re.search(r"\d+", h_text)
                    or any(w in h_lower for w in [
                        "best", "top", "why", "secret", "ultimate",
                        "guide", "tips", "ways", "mistakes",
                    ])
                ):
                    engaging_heading_count += 1

            details["engaging_headings"] = engaging_heading_count
            if engaging_heading_count >= 3:
                score += 0.3

        return DimensionScore(
            dimension=QualityDimension.ENGAGEMENT,
            score=_clamp(score),
            details=details,
            suggestions=suggestions,
        )

    async def _score_voice_match(
        self, text: str, site_id: str
    ) -> DimensionScore:
        """Score content against the site's brand voice profile.

        Lazy-imports BrandVoiceEngine to avoid circular dependencies.
        Falls back to a neutral score (7.0) if the engine is unavailable.
        """
        details: Dict[str, Any] = {}
        suggestions: List[str] = []

        try:
            from src.brand_voice_engine import BrandVoiceEngine, VoiceScore

            engine = BrandVoiceEngine()
            voice_score: VoiceScore = engine.score_content(text, site_id)

            # VoiceScore has: overall (0-10), tone_match, vocabulary_match,
            # rule_compliance, feedback
            raw_score = getattr(voice_score, "overall", None)
            if raw_score is None:
                raw_score = getattr(voice_score, "score", 7.0)

            score = float(raw_score)
            details["voice_engine_available"] = True
            details["voice_overall"] = round(score, 2)

            tone_match = getattr(voice_score, "tone_match", None)
            if tone_match is not None:
                details["tone_match"] = round(float(tone_match), 2)

            vocab_match = getattr(voice_score, "vocabulary_match", None)
            if vocab_match is not None:
                details["vocabulary_match"] = round(float(vocab_match), 2)

            feedback = getattr(voice_score, "feedback", None)
            if feedback and isinstance(feedback, str):
                suggestions.append(f"Voice feedback: {feedback[:200]}")
            elif feedback and isinstance(feedback, list):
                for fb in feedback[:3]:
                    suggestions.append(f"Voice: {fb[:150]}")

            return DimensionScore(
                dimension=QualityDimension.VOICE_MATCH,
                score=_clamp(score),
                details=details,
                suggestions=suggestions,
            )

        except ImportError:
            logger.debug(
                "BrandVoiceEngine not available; using neutral voice score"
            )
            details["voice_engine_available"] = False
            return DimensionScore(
                dimension=QualityDimension.VOICE_MATCH,
                score=7.0,
                details=details,
                suggestions=[
                    "BrandVoiceEngine unavailable. Install brand_voice_engine "
                    "module for voice adherence scoring."
                ],
            )
        except Exception as exc:
            logger.warning("Voice scoring failed: %s", exc)
            details["voice_engine_available"] = True
            details["voice_error"] = str(exc)
            return DimensionScore(
                dimension=QualityDimension.VOICE_MATCH,
                score=7.0,
                details=details,
                suggestions=[
                    f"Voice scoring encountered an error: {str(exc)[:100]}. "
                    f"Score defaulted to 7.0."
                ],
            )

    # ------------------------------------------------------------------
    # Quality Gates
    # ------------------------------------------------------------------

    def check_gate(
        self, report: QualityReport
    ) -> Tuple[bool, List[str]]:
        """Check whether a quality report passes the gate.

        Returns:
            (passed, reasons): Boolean pass/fail and a list of failure reasons.
        """
        reasons: List[str] = []

        if report.overall_score < self._threshold:
            reasons.append(
                f"Overall score {report.overall_score:.1f} is below "
                f"threshold {self._threshold:.1f}."
            )

        # Per-dimension minimum (any dimension below 3.0 auto-fails)
        for dim, ds in report.dimensions.items():
            if ds.score < 3.0:
                reasons.append(
                    f"Dimension '{dim.value}' scored {ds.score:.1f}/10 "
                    f"(minimum 3.0 required)."
                )

        # Critical content length gate
        if report.word_count < 100:
            reasons.append(
                f"Content is only {report.word_count} words. "
                f"Minimum 100 words required."
            )

        # High duplicate score gate
        if report.duplicate_score > 0.70:
            reasons.append(
                f"Duplicate score is {report.duplicate_score:.0%}. "
                f"Content appears to be a near-duplicate."
            )

        passed = len(reasons) == 0
        return passed, reasons

    def set_threshold(self, score: float) -> None:
        """Update the minimum passing score for quality gates."""
        self._threshold = _clamp(score, 0.0, 10.0)
        self._save_config()
        logger.info("Quality threshold updated to %.1f", self._threshold)

    def set_weights(
        self, weights: Dict[str, float]
    ) -> None:
        """Update dimension weights.

        Args:
            weights: Dict mapping dimension name to weight.
                     Weights are normalized to sum to 1.0.
        """
        for key, value in weights.items():
            try:
                dim = QualityDimension(key)
                self._dimension_weights[dim] = max(0.0, float(value))
            except ValueError:
                logger.warning("Unknown dimension '%s' -- skipping.", key)

        # Normalize
        total = sum(self._dimension_weights.values())
        if total > 0:
            for dim in self._dimension_weights:
                self._dimension_weights[dim] /= total

        self._save_config()
        logger.info(
            "Dimension weights updated: %s",
            {d.value: round(w, 3) for d, w in self._dimension_weights.items()},
        )

    # ------------------------------------------------------------------
    # Duplicate Detection
    # ------------------------------------------------------------------

    def _build_fingerprint(self, content: str) -> List[str]:
        """Build a shingle-based fingerprint for content.

        Returns a list of hash strings for JSON serialization.
        """
        shingles = _build_shingles(content)
        return _fingerprint_to_list(shingles)

    def _check_duplicates(
        self, shingles: Set[str], site_id: str
    ) -> Tuple[float, Optional[str]]:
        """Check shingles against the corpus for the given site.

        Returns:
            (max_jaccard, closest_article_id): Highest similarity and the
            ID of the most similar article.
        """
        corpus_file = CORPUS_DIR / f"{site_id}.json"
        corpus = _load_json(corpus_file, [])

        if not corpus:
            return 0.0, None

        fingerprint_hashes = _fingerprint_to_list(shingles)
        fp_set = set(fingerprint_hashes)

        max_sim = 0.0
        closest_id: Optional[str] = None

        for entry in corpus:
            stored_hashes = set(entry.get("fingerprint", []))
            if not stored_hashes:
                continue

            sim = _jaccard_similarity(fp_set, stored_hashes)
            if sim > max_sim:
                max_sim = sim
                closest_id = entry.get("article_id")

        return max_sim, closest_id

    def add_to_corpus(
        self, content: str, site_id: str, article_id: str
    ) -> None:
        """Store a content fingerprint in the site's corpus.

        Args:
            content: The article text (HTML or plain).
            site_id: The target site identifier.
            article_id: Unique article identifier.
        """
        plain_text, _, _, _ = _strip_html(content)
        fingerprint = self._build_fingerprint(plain_text)

        corpus_file = CORPUS_DIR / f"{site_id}.json"
        corpus = _load_json(corpus_file, [])

        # Avoid duplicating the same article_id
        corpus = [e for e in corpus if e.get("article_id") != article_id]

        corpus.append({
            "article_id": article_id,
            "fingerprint": fingerprint,
            "added_at": _now_iso(),
        })

        # Keep corpus bounded (last 500 articles per site)
        if len(corpus) > 500:
            corpus = corpus[-500:]

        _save_json(corpus_file, corpus)
        logger.debug(
            "Added fingerprint for article '%s' to site '%s' corpus "
            "(%d entries).",
            article_id[:12], site_id, len(corpus),
        )

    # ------------------------------------------------------------------
    # History & Analytics
    # ------------------------------------------------------------------

    def _append_history(self, report: QualityReport) -> None:
        """Append a report to scoring history (bounded at MAX_HISTORY)."""
        entry = report.to_dict()
        self._history.append(entry)

        # Trim if over limit
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

        _save_json(HISTORY_FILE, self._history)

    def get_history(
        self,
        site_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get scoring history, optionally filtered by site.

        Args:
            site_id: Filter by site. All sites if None.
            limit: Maximum entries to return (newest first).

        Returns:
            List of report dicts, newest first.
        """
        history = list(reversed(self._history))

        if site_id:
            history = [h for h in history if h.get("site_id") == site_id]

        return history[:limit]

    def get_average_scores(
        self,
        site_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get average scores across dimensions over a time period.

        Args:
            site_id: Filter by site. All sites if None.
            days: Look-back window in days.

        Returns:
            Dict with overall and per-dimension averages.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        filtered = []
        for h in self._history:
            scored_at = h.get("scored_at", "")
            if scored_at < cutoff:
                continue
            if site_id and h.get("site_id") != site_id:
                continue
            filtered.append(h)

        if not filtered:
            return {
                "count": 0,
                "overall_avg": 0.0,
                "dimensions": {},
                "pass_rate": 0.0,
            }

        overall_scores = [h.get("overall_score", 0.0) for h in filtered]
        overall_avg = sum(overall_scores) / len(overall_scores)

        # Per-dimension averages
        dim_totals: Dict[str, List[float]] = {}
        for h in filtered:
            dims = h.get("dimensions", {})
            for dim_name, dim_data in dims.items():
                if dim_name not in dim_totals:
                    dim_totals[dim_name] = []
                dim_totals[dim_name].append(dim_data.get("score", 0.0))

        dim_avgs = {}
        for dim_name, scores in dim_totals.items():
            dim_avgs[dim_name] = round(sum(scores) / len(scores), 2)

        # Pass rate
        passed_count = sum(1 for h in filtered if h.get("passed", False))
        pass_rate = passed_count / len(filtered) if filtered else 0.0

        return {
            "count": len(filtered),
            "overall_avg": round(overall_avg, 2),
            "dimensions": dim_avgs,
            "pass_rate": round(pass_rate, 3),
            "period_days": days,
            "site_id": site_id or "all",
        }

    def get_quality_trend(
        self,
        site_id: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict]:
        """Get daily quality score trend over a time period.

        Args:
            site_id: Filter by site. All sites if None.
            days: Look-back window in days.

        Returns:
            List of dicts with date and average score, sorted by date.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        daily: Dict[str, List[float]] = {}
        for h in self._history:
            scored_at = h.get("scored_at", "")
            if scored_at < cutoff:
                continue
            if site_id and h.get("site_id") != site_id:
                continue

            date_str = scored_at[:10]  # YYYY-MM-DD
            if date_str not in daily:
                daily[date_str] = []
            daily[date_str].append(h.get("overall_score", 0.0))

        trend = []
        for date_str in sorted(daily.keys()):
            scores = daily[date_str]
            trend.append({
                "date": date_str,
                "avg_score": round(sum(scores) / len(scores), 2),
                "count": len(scores),
                "min_score": round(min(scores), 2),
                "max_score": round(max(scores), 2),
            })

        return trend

    def get_failing_articles(self, days: int = 7) -> List[Dict]:
        """Get articles that failed the quality gate recently.

        Args:
            days: Look-back window in days.

        Returns:
            List of failing report summaries, newest first.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        failing = []
        for h in reversed(self._history):
            scored_at = h.get("scored_at", "")
            if scored_at < cutoff:
                continue
            if not h.get("passed", True):
                failing.append({
                    "article_id": h.get("article_id", "?")[:12],
                    "site_id": h.get("site_id", "?"),
                    "title": h.get("title", "(unknown)")[:60],
                    "overall_score": h.get("overall_score", 0.0),
                    "grade": h.get("grade", "?"),
                    "scored_at": scored_at,
                    "top_suggestions": h.get("top_suggestions", [])[:3],
                })

        return failing

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate scoring statistics.

        Returns:
            Dict with total counts, grade distribution, pass rate, etc.
        """
        if not self._history:
            return {
                "total_scored": 0,
                "by_site": {},
                "by_grade": {},
                "overall_avg": 0.0,
                "pass_rate": 0.0,
                "recent_7_days": 0,
                "recent_30_days": 0,
                "threshold": self._threshold,
            }

        now_str = _now_iso()
        seven_ago = (
            datetime.now(timezone.utc) - timedelta(days=7)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        thirty_ago = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        by_site: Dict[str, int] = {}
        by_grade: Dict[str, int] = {}
        scores: List[float] = []
        passed_count = 0
        recent_7 = 0
        recent_30 = 0

        for h in self._history:
            site = h.get("site_id", "unknown")
            by_site[site] = by_site.get(site, 0) + 1

            grade = h.get("grade", "?")
            by_grade[grade] = by_grade.get(grade, 0) + 1

            scores.append(h.get("overall_score", 0.0))

            if h.get("passed", False):
                passed_count += 1

            scored_at = h.get("scored_at", "")
            if scored_at >= seven_ago:
                recent_7 += 1
            if scored_at >= thirty_ago:
                recent_30 += 1

        overall_avg = sum(scores) / len(scores) if scores else 0.0
        pass_rate = (
            passed_count / len(self._history) if self._history else 0.0
        )

        return {
            "total_scored": len(self._history),
            "by_site": dict(sorted(by_site.items(), key=lambda x: -x[1])),
            "by_grade": dict(sorted(by_grade.items())),
            "overall_avg": round(overall_avg, 2),
            "pass_rate": round(pass_rate, 3),
            "recent_7_days": recent_7,
            "recent_30_days": recent_30,
            "threshold": self._threshold,
        }

    # ------------------------------------------------------------------
    # Config Persistence
    # ------------------------------------------------------------------

    def _save_config(self) -> None:
        """Persist threshold and weights to config file."""
        config = {
            "threshold": self._threshold,
            "weights": {
                dim.value: round(w, 4)
                for dim, w in self._dimension_weights.items()
            },
            "updated_at": _now_iso(),
        }
        _save_json(CONFIG_FILE, config)


# ---------------------------------------------------------------------------
# Sync Runner (defined above via _run_sync)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_scorer: Optional[ContentQualityScorer] = None


def get_scorer() -> ContentQualityScorer:
    """Get or create the singleton ContentQualityScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ContentQualityScorer()
    return _scorer


# ---------------------------------------------------------------------------
# CLI Handlers
# ---------------------------------------------------------------------------


def _cli_score(args: argparse.Namespace) -> None:
    """Handle the 'score' CLI command."""
    scorer = get_scorer()

    # Read content from file or stdin
    if args.stdin:
        content = sys.stdin.read()
        if not content.strip():
            print("Error: No content received on stdin.", file=sys.stderr)
            sys.exit(1)
    elif args.content:
        content_path = Path(args.content)
        if not content_path.exists():
            print(
                f"Error: File not found: {content_path}", file=sys.stderr
            )
            sys.exit(1)
        content = content_path.read_text(encoding="utf-8", errors="replace")
    else:
        print(
            "Error: Provide --content FILE or --stdin.", file=sys.stderr
        )
        sys.exit(1)

    keywords = []
    if args.keywords:
        keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]

    print(f"\nScoring content for site '{args.site}'...\n")

    try:
        report = scorer.score_sync(
            content=content,
            title=args.title,
            site_id=args.site,
            keywords=keywords,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(report.summary())

    # Add to corpus if requested
    if args.add_to_corpus:
        scorer.add_to_corpus(content, args.site, report.article_id)
        print(f"\n  Fingerprint added to '{args.site}' corpus.\n")

    # Output JSON if requested
    if args.json:
        print("\n--- JSON Report ---")
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


def _cli_history(args: argparse.Namespace) -> None:
    """Handle the 'history' CLI command."""
    scorer = get_scorer()
    site_filter = args.site if args.site != "all" else None
    results = scorer.get_history(site_id=site_filter, limit=args.limit)

    if not results:
        label = f"site '{site_filter}'" if site_filter else "any site"
        print(f"\nNo scoring history found for {label}.\n")
        return

    print(f"\n{'=' * 78}")
    header = "QUALITY SCORING HISTORY"
    if site_filter:
        header += f" (site: {site_filter})"
    print(f"  {header}")
    print(f"{'=' * 78}\n")

    print(
        f"  {'ID':<14} {'Site':<18} {'Title':<28} "
        f"{'Score':>6} {'Grade':>6} {'Gate':>6} {'Date':<12}"
    )
    print(
        f"  {'-' * 14} {'-' * 18} {'-' * 28} "
        f"{'-' * 6} {'-' * 6} {'-' * 6} {'-' * 12}"
    )

    for r in results:
        title = r.get("title", "(unknown)")
        if len(title) > 26:
            title = title[:23] + "..."
        aid = r.get("article_id", "?")[:12]
        date_str = r.get("scored_at", "")[:10]
        gate = "PASS" if r.get("passed") else "FAIL"
        print(
            f"  {aid:<14} {r.get('site_id', '?'):<18} {title:<28} "
            f"{r.get('overall_score', 0):.1f:>6} "
            f"{r.get('grade', '?'):>6} {gate:>6} {date_str:<12}"
        )

    print(f"\n  Total: {len(results)} entry(ies)")
    print(f"{'=' * 78}\n")


def _cli_trend(args: argparse.Namespace) -> None:
    """Handle the 'trend' CLI command."""
    scorer = get_scorer()
    site_filter = args.site if args.site != "all" else None
    trend = scorer.get_quality_trend(site_id=site_filter, days=args.days)

    if not trend:
        print(f"\nNo trend data available for the last {args.days} days.\n")
        return

    print(f"\n{'=' * 60}")
    header = f"QUALITY TREND (last {args.days} days)"
    if site_filter:
        header += f" -- site: {site_filter}"
    print(f"  {header}")
    print(f"{'=' * 60}\n")

    print(
        f"  {'Date':<12} {'Avg':>6} {'Min':>6} {'Max':>6} {'Count':>6}"
    )
    print(
        f"  {'-' * 12} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}"
    )

    for t in trend:
        bar = "#" * int(t["avg_score"]) + "-" * (10 - int(t["avg_score"]))
        print(
            f"  {t['date']:<12} {t['avg_score']:>5.1f} "
            f"{t['min_score']:>6.1f} {t['max_score']:>6.1f} "
            f"{t['count']:>6}  [{bar}]"
        )

    # Overall summary
    all_avgs = [t["avg_score"] for t in trend]
    period_avg = sum(all_avgs) / len(all_avgs) if all_avgs else 0.0
    total_count = sum(t["count"] for t in trend)

    print(f"\n  Period Average: {period_avg:.1f}/10")
    print(f"  Total Articles: {total_count}")
    print(f"{'=' * 60}\n")


def _cli_stats(args: argparse.Namespace) -> None:
    """Handle the 'stats' CLI command."""
    scorer = get_scorer()
    stats = scorer.get_stats()

    print(f"\n{'=' * 60}")
    print(f"  CONTENT QUALITY SCORING STATISTICS")
    print(f"{'=' * 60}\n")

    print(f"  Total Scored:      {stats['total_scored']:,}")
    print(f"  Overall Average:   {stats['overall_avg']:.1f}/10")
    print(f"  Pass Rate:         {stats['pass_rate']:.1%}")
    print(f"  Threshold:         {stats['threshold']:.1f}")
    print(f"  Last 7 Days:       {stats['recent_7_days']:,}")
    print(f"  Last 30 Days:      {stats['recent_30_days']:,}")
    print()

    if stats["by_grade"]:
        print(f"  By Grade:")
        for grade, count in stats["by_grade"].items():
            print(f"    {grade:<6} {count:>5} article(s)")
        print()

    if stats["by_site"]:
        print(f"  By Site:")
        for site, count in stats["by_site"].items():
            print(f"    {site:<25} {count:>5} scored")
        print()

    print(f"{'=' * 60}\n")


def _cli_threshold(args: argparse.Namespace) -> None:
    """Handle the 'threshold' CLI command."""
    scorer = get_scorer()

    if args.score is not None:
        scorer.set_threshold(args.score)
        print(f"\nQuality gate threshold set to {args.score:.1f}/10.\n")
    else:
        print(f"\nCurrent threshold: {scorer._threshold:.1f}/10\n")


def _cli_failing(args: argparse.Namespace) -> None:
    """Handle the 'failing' CLI command."""
    scorer = get_scorer()
    failing = scorer.get_failing_articles(days=args.days)

    if not failing:
        print(f"\nNo failing articles in the last {args.days} day(s).\n")
        return

    print(f"\n{'=' * 78}")
    print(f"  FAILING ARTICLES (last {args.days} days)")
    print(f"{'=' * 78}\n")

    for f in failing:
        print(f"  [{f['grade']}] {f['overall_score']:.1f}/10 -- "
              f"{f['site_id']}: {f['title']}")
        print(f"      ID: {f['article_id']}  Scored: {f['scored_at'][:10]}")
        if f.get("top_suggestions"):
            for s in f["top_suggestions"]:
                print(f"      -> {s[:70]}")
        print()

    print(f"  Total failing: {len(failing)} article(s)")
    print(f"{'=' * 78}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the Content Quality Scorer."""
    parser = argparse.ArgumentParser(
        prog="content_quality_scorer",
        description=(
            "Content Quality Scorer for the OpenClaw Empire. Scores article "
            "content across 7 quality dimensions (readability, E-E-A-T, SEO, "
            "structure, originality, engagement, voice match) and enforces "
            "quality gates before publication."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- score ---
    sub_score = subparsers.add_parser(
        "score",
        help="Score article content for quality",
    )
    sub_score.add_argument(
        "--content",
        help="Path to content file (HTML or plain text)",
    )
    sub_score.add_argument(
        "--stdin",
        action="store_true",
        help="Read content from stdin",
    )
    sub_score.add_argument(
        "--title",
        required=True,
        help="Article title / headline",
    )
    sub_score.add_argument(
        "--site",
        required=True,
        help=f"Site ID ({', '.join(VALID_SITE_IDS[:5])}...)",
    )
    sub_score.add_argument(
        "--keywords",
        help="Comma-separated focus keywords",
    )
    sub_score.add_argument(
        "--add-to-corpus",
        action="store_true",
        help="Add article fingerprint to duplicate detection corpus",
    )
    sub_score.add_argument(
        "--json",
        action="store_true",
        help="Also output full JSON report",
    )
    sub_score.set_defaults(func=_cli_score)

    # --- history ---
    sub_history = subparsers.add_parser(
        "history",
        help="Show scoring history",
    )
    sub_history.add_argument(
        "--site",
        default="all",
        help="Filter by site ID (default: all)",
    )
    sub_history.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum entries to show (default: 50)",
    )
    sub_history.set_defaults(func=_cli_history)

    # --- trend ---
    sub_trend = subparsers.add_parser(
        "trend",
        help="Show quality score trend over time",
    )
    sub_trend.add_argument(
        "--site",
        default="all",
        help="Filter by site ID (default: all)",
    )
    sub_trend.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look-back window in days (default: 30)",
    )
    sub_trend.set_defaults(func=_cli_trend)

    # --- stats ---
    sub_stats = subparsers.add_parser(
        "stats",
        help="Show aggregate scoring statistics",
    )
    sub_stats.set_defaults(func=_cli_stats)

    # --- threshold ---
    sub_threshold = subparsers.add_parser(
        "threshold",
        help="View or update quality gate threshold",
    )
    sub_threshold.add_argument(
        "--score",
        type=float,
        help="New threshold score (0-10)",
    )
    sub_threshold.set_defaults(func=_cli_threshold)

    # --- failing ---
    sub_failing = subparsers.add_parser(
        "failing",
        help="Show articles that failed the quality gate",
    )
    sub_failing.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look-back window in days (default: 7)",
    )
    sub_failing.set_defaults(func=_cli_failing)

    # Parse and dispatch
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
