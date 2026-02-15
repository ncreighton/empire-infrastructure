"""Test content_quality_scorer -- OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# We patch the data directory before importing the module so it does not
# touch real project data during test runs.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_data_dirs(tmp_path, monkeypatch):
    """Redirect all scorer file I/O to a temp directory."""
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "corpus").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.content_quality_scorer.QUALITY_DATA_DIR", quality_dir,
    )
    monkeypatch.setattr(
        "src.content_quality_scorer.CORPUS_DIR", quality_dir / "corpus",
    )
    monkeypatch.setattr(
        "src.content_quality_scorer.HISTORY_FILE", quality_dir / "history.json",
    )
    monkeypatch.setattr(
        "src.content_quality_scorer.CONFIG_FILE", quality_dir / "config.json",
    )


# ---------------------------------------------------------------------------
# Imports (after fixtures defined so monkeypatching can intercept)
# ---------------------------------------------------------------------------

from src.content_quality_scorer import (
    ContentQualityScorer,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    DimensionScore,
    FLAGGED_PHRASES,
    QualityDimension,
    QualityGrade,
    QualityReport,
    VALID_SITE_IDS,
    _build_shingles,
    _clamp,
    _count_syllables,
    _flesch_kincaid_grade,
    _flesch_reading_ease,
    _jaccard_similarity,
    _reading_level_from_grade,
    _score_to_grade,
    _split_paragraphs,
    _split_sentences,
    _split_words,
    _strip_html,
    get_scorer,
)


# ===========================================================================
# QUALITY DIMENSION ENUM
# ===========================================================================


class TestQualityDimension:
    """Verify the seven quality dimensions are defined correctly."""

    def test_seven_dimensions_exist(self):
        dims = list(QualityDimension)
        assert len(dims) == 7

    def test_dimension_values(self):
        expected = {
            "readability", "eeat", "seo", "structure",
            "originality", "engagement", "voice_match",
        }
        assert {d.value for d in QualityDimension} == expected

    def test_dimension_is_str_enum(self):
        assert isinstance(QualityDimension.READABILITY, str)
        assert QualityDimension.SEO == "seo"


# ===========================================================================
# QUALITY GRADE ENUM
# ===========================================================================


class TestQualityGrade:
    """Verify grade enum members."""

    def test_six_grades(self):
        assert len(list(QualityGrade)) == 6

    def test_grade_values(self):
        assert QualityGrade.A_PLUS.value == "A+"
        assert QualityGrade.F.value == "F"


# ===========================================================================
# SCORE TO GRADE
# ===========================================================================


class TestScoreToGrade:
    """Test numeric score to letter grade mapping."""

    @pytest.mark.parametrize("score,expected", [
        (10.0, QualityGrade.A_PLUS),
        (9.0, QualityGrade.A_PLUS),
        (8.5, QualityGrade.A),
        (8.0, QualityGrade.A),
        (7.0, QualityGrade.B),
        (6.0, QualityGrade.C),
        (5.5, QualityGrade.D),
        (5.0, QualityGrade.D),
        (4.9, QualityGrade.F),
        (0.0, QualityGrade.F),
    ])
    def test_score_grade_boundaries(self, score, expected):
        assert _score_to_grade(score) == expected


# ===========================================================================
# CLAMP HELPER
# ===========================================================================


class TestClamp:
    """Test value clamping to [lo, hi] range."""

    def test_within_range(self):
        assert _clamp(5.0) == 5.0

    def test_below_lo(self):
        assert _clamp(-1.0) == 0.0

    def test_above_hi(self):
        assert _clamp(15.0) == 10.0

    def test_custom_bounds(self):
        assert _clamp(50.0, lo=0, hi=100) == 50.0
        assert _clamp(-5.0, lo=0, hi=100) == 0.0


# ===========================================================================
# SYLLABLE COUNTING
# ===========================================================================


class TestCountSyllables:
    """Test vowel-group syllable counting heuristic."""

    def test_one_syllable_words(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("the") == 1

    def test_multi_syllable(self):
        assert _count_syllables("beautiful") >= 3
        assert _count_syllables("information") >= 3

    def test_silent_e(self):
        # "cake" should count as 1 syllable (silent e)
        assert _count_syllables("cake") == 1

    def test_empty_word(self):
        assert _count_syllables("") == 0

    def test_non_alpha_stripped(self):
        assert _count_syllables("it's") >= 1

    def test_minimum_one(self):
        # Every non-empty word should have at least 1 syllable
        assert _count_syllables("fly") >= 1


# ===========================================================================
# FLESCH-KINCAID
# ===========================================================================


class TestFleschKincaid:
    """Test Flesch-Kincaid grade level and reading ease calculations."""

    def test_grade_zero_for_no_words(self):
        assert _flesch_kincaid_grade(0, 0, 0) == 0.0

    def test_grade_zero_for_no_sentences(self):
        assert _flesch_kincaid_grade(100, 0, 200) == 0.0

    def test_simple_grade_calculation(self):
        # 100 words, 10 sentences, 120 syllables
        grade = _flesch_kincaid_grade(100, 10, 120)
        assert grade > 0.0
        assert isinstance(grade, float)

    def test_reading_ease_zero_for_empty(self):
        assert _flesch_reading_ease(0, 0, 0) == 0.0

    def test_reading_ease_in_valid_range(self):
        ease = _flesch_reading_ease(100, 10, 120)
        assert 0.0 <= ease <= 100.0


# ===========================================================================
# READING LEVEL FROM GRADE
# ===========================================================================


class TestReadingLevelFromGrade:

    @pytest.mark.parametrize("grade,level", [
        (4.0, "easy"),
        (6.0, "easy"),
        (8.0, "medium"),
        (10.0, "medium"),
        (12.0, "hard"),
        (14.0, "hard"),
        (16.0, "expert"),
    ])
    def test_level_boundaries(self, grade, level):
        assert _reading_level_from_grade(grade) == level


# ===========================================================================
# TEXT SPLITTING
# ===========================================================================


class TestTextSplitting:
    """Test sentence, word, and paragraph splitting."""

    def test_split_sentences_basic(self):
        text = "Hello world. How are you? I am fine!"
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_split_sentences_handles_abbreviations(self):
        text = "Dr. Smith went home. He was tired."
        sentences = _split_sentences(text)
        # "Dr." should NOT trigger a split
        assert len(sentences) == 2

    def test_split_words(self):
        words = _split_words("Hello world, it's a beautiful day!")
        assert "Hello" in words
        assert "world" in words

    def test_split_words_empty(self):
        assert _split_words("") == []

    def test_split_paragraphs(self):
        text = "Para one line one.\nPara one line two.\n\nPara two."
        paragraphs = _split_paragraphs(text)
        assert len(paragraphs) == 2


# ===========================================================================
# HTML STRIPPING
# ===========================================================================


class TestStripHtml:
    """Test HTML tag removal and heading extraction."""

    def test_strip_simple_html(self):
        html = "<p>Hello <b>world</b></p>"
        text, headings, list_count, img_count = _strip_html(html)
        assert "Hello" in text
        assert "world" in text
        assert "<" not in text

    def test_extract_headings(self):
        html = "<h2>Introduction</h2><p>Text here</p><h3>Sub</h3>"
        _, headings, _, _ = _strip_html(html)
        assert len(headings) == 2
        assert headings[0] == ("h2", "Introduction")
        assert headings[1] == ("h3", "Sub")

    def test_count_lists(self):
        html = "<ul><li>One</li></ul><ol><li>Two</li></ol>"
        _, _, list_count, _ = _strip_html(html)
        assert list_count == 2

    def test_count_images(self):
        html = '<p>Text</p><img src="a.jpg"/><img src="b.png"/>'
        _, _, _, img_count = _strip_html(html)
        assert img_count == 2


# ===========================================================================
# SHINGLE-BASED DUPLICATE DETECTION
# ===========================================================================


class TestShingleDuplicateDetection:
    """Test shingle construction and Jaccard similarity."""

    def test_build_shingles_basic(self):
        text = "the quick brown fox jumps over the lazy dog"
        shingles = _build_shingles(text, size=3)
        assert len(shingles) > 0

    def test_build_shingles_short_text(self):
        shingles = _build_shingles("hi there", size=5)
        # Text shorter than shingle size should still return something
        assert len(shingles) >= 1

    def test_build_shingles_empty(self):
        shingles = _build_shingles("", size=5)
        assert len(shingles) == 0

    def test_jaccard_identical(self):
        s = {"a b c", "b c d", "c d e"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint(self):
        a = {"x y z"}
        b = {"a b c"}
        assert _jaccard_similarity(a, b) == 0.0

    def test_jaccard_empty_set(self):
        assert _jaccard_similarity(set(), {"a b c"}) == 0.0

    def test_jaccard_partial_overlap(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        sim = _jaccard_similarity(a, b)
        # intersection=2, union=4 -> 0.5
        assert abs(sim - 0.5) < 0.01


# ===========================================================================
# DIMENSION SCORE DATACLASS
# ===========================================================================


class TestDimensionScore:

    def test_to_dict(self):
        ds = DimensionScore(
            dimension=QualityDimension.READABILITY,
            score=7.5,
            details={"fk_grade": 8.0},
            suggestions=["Shorten sentences"],
        )
        d = ds.to_dict()
        assert d["dimension"] == "readability"
        assert d["score"] == 7.5
        assert "fk_grade" in d["details"]

    def test_score_rounding(self):
        ds = DimensionScore(
            dimension=QualityDimension.SEO,
            score=7.333333,
        )
        assert ds.to_dict()["score"] == 7.33


# ===========================================================================
# QUALITY REPORT DATACLASS
# ===========================================================================


class TestQualityReport:

    def test_to_dict_structure(self):
        report = QualityReport(
            article_id="test-id",
            site_id="witchcraft",
            title="Test Article",
            overall_score=7.5,
            grade=QualityGrade.B,
            passed=True,
            word_count=1500,
            sentence_count=60,
        )
        d = report.to_dict()
        assert d["article_id"] == "test-id"
        assert d["grade"] == "B"
        assert d["passed"] is True

    def test_summary_returns_string(self):
        report = QualityReport(
            article_id="abc",
            site_id="witchcraft",
            title="Moon Water",
            overall_score=8.0,
            grade=QualityGrade.A,
            passed=True,
            word_count=2000,
            sentence_count=80,
            paragraph_count=15,
            avg_sentence_length=25.0,
            reading_level="medium",
            estimated_reading_time_minutes=8.4,
            duplicate_score=0.05,
        )
        summary = report.summary()
        assert "CONTENT QUALITY REPORT" in summary
        assert "Moon Water" in summary
        assert "PASSED" in summary


# ===========================================================================
# ContentQualityScorer INITIALIZATION
# ===========================================================================


class TestScorerInit:
    """Test scorer initialization and configuration."""

    def test_default_threshold(self):
        scorer = ContentQualityScorer()
        assert scorer._threshold == DEFAULT_THRESHOLD

    def test_custom_threshold(self):
        scorer = ContentQualityScorer(threshold=7.5)
        assert scorer._threshold == 7.5

    def test_weights_sum_to_one(self):
        scorer = ContentQualityScorer()
        total = sum(scorer._dimension_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        custom = {"readability": 0.5, "seo": 0.5}
        scorer = ContentQualityScorer(weights=custom)
        assert scorer._dimension_weights[QualityDimension.READABILITY] > 0
        assert scorer._dimension_weights[QualityDimension.SEO] > 0


# ===========================================================================
# ContentQualityScorer.score()
# ===========================================================================


class TestScorerScore:
    """Test the main scoring method across all 7 dimensions."""

    @pytest.fixture
    def scorer(self):
        return ContentQualityScorer()

    @pytest.fixture
    def sample_html(self):
        return (
            "<h2>Introduction to Moon Water</h2>"
            "<p>Moon water is water that has been charged under the light of "
            "the full moon. I've been making moon water for years and I can "
            "tell you it truly works. Research shows that intention-setting "
            "rituals can improve mindfulness.</p>"
            "<h3>How to Make Moon Water</h3>"
            "<p>Fill a clean glass jar with spring water. Place it where the "
            "moonlight can reach it. Leave it overnight. In the morning, your "
            "moon water is ready to use.</p>"
            "<h3>Uses for Moon Water</h3>"
            "<ul><li>Spiritual cleansing</li><li>Watering plants</li>"
            "<li>Adding to baths</li></ul>"
            "<p>Disclaimer: This is not medical advice. Always consult with a "
            "qualified practitioner for health-related matters.</p>"
            "<p>Sign up for our newsletter to learn more about moon rituals!</p>"
        )

    @pytest.mark.asyncio
    async def test_score_returns_quality_report(self, scorer, sample_html):
        report = await scorer.score(
            content=sample_html,
            title="Moon Water Guide for Beginners",
            site_id="witchcraft",
            keywords=["moon water", "ritual"],
        )
        assert isinstance(report, QualityReport)
        assert report.site_id == "witchcraft"
        assert report.title == "Moon Water Guide for Beginners"

    @pytest.mark.asyncio
    async def test_score_has_all_seven_dimensions(self, scorer, sample_html):
        report = await scorer.score(
            content=sample_html,
            title="Moon Water",
            site_id="witchcraft",
        )
        for dim in QualityDimension:
            assert dim in report.dimensions, f"Missing dimension: {dim}"
            ds = report.dimensions[dim]
            assert 0.0 <= ds.score <= 10.0

    @pytest.mark.asyncio
    async def test_overall_score_in_valid_range(self, scorer, sample_html):
        report = await scorer.score(
            content=sample_html,
            title="Test",
            site_id="witchcraft",
        )
        assert 0.0 <= report.overall_score <= 10.0

    @pytest.mark.asyncio
    async def test_word_count_populated(self, scorer, sample_html):
        report = await scorer.score(
            content=sample_html,
            title="Test",
            site_id="witchcraft",
        )
        assert report.word_count > 0
        assert report.sentence_count > 0
        assert report.paragraph_count > 0

    @pytest.mark.asyncio
    async def test_score_assigns_grade(self, scorer, sample_html):
        report = await scorer.score(
            content=sample_html,
            title="Test",
            site_id="witchcraft",
        )
        assert isinstance(report.grade, QualityGrade)

    @pytest.mark.asyncio
    async def test_empty_content_scores_low(self, scorer):
        report = await scorer.score(
            content="<p>Short.</p>",
            title="Very Short Article",
            site_id="witchcraft",
        )
        assert report.word_count < 10
        assert report.overall_score < 7.0


# ===========================================================================
# ContentQualityScorer.check_gate()
# ===========================================================================


class TestCheckGate:
    """Test quality gate thresholds and failure reasons."""

    @pytest.fixture
    def scorer(self):
        return ContentQualityScorer(threshold=6.0)

    def test_passes_when_above_threshold(self, scorer):
        report = QualityReport(
            article_id="test",
            site_id="witchcraft",
            title="Good Article",
            overall_score=7.5,
            grade=QualityGrade.B,
            passed=True,
            word_count=2000,
        )
        passed, reasons = scorer.check_gate(report)
        assert passed is True
        assert len(reasons) == 0

    def test_fails_when_below_threshold(self, scorer):
        report = QualityReport(
            article_id="test",
            site_id="witchcraft",
            title="Bad Article",
            overall_score=4.5,
            grade=QualityGrade.F,
            passed=False,
            word_count=2000,
        )
        passed, reasons = scorer.check_gate(report)
        assert passed is False
        assert any("below threshold" in r for r in reasons)

    def test_fails_with_low_dimension_score(self, scorer):
        report = QualityReport(
            article_id="test",
            site_id="witchcraft",
            title="Partial Fail",
            overall_score=7.0,
            grade=QualityGrade.B,
            passed=True,
            word_count=2000,
            dimensions={
                QualityDimension.READABILITY: DimensionScore(
                    dimension=QualityDimension.READABILITY,
                    score=2.0,  # Below 3.0 minimum
                ),
            },
        )
        passed, reasons = scorer.check_gate(report)
        assert passed is False
        assert any("minimum 3.0" in r for r in reasons)

    def test_fails_on_too_few_words(self, scorer):
        report = QualityReport(
            article_id="test",
            site_id="witchcraft",
            title="Tiny",
            overall_score=7.0,
            grade=QualityGrade.B,
            passed=True,
            word_count=50,
        )
        passed, reasons = scorer.check_gate(report)
        assert passed is False
        assert any("100 words" in r for r in reasons)

    def test_fails_on_high_duplicate_score(self, scorer):
        report = QualityReport(
            article_id="test",
            site_id="witchcraft",
            title="Duplicate",
            overall_score=7.0,
            grade=QualityGrade.B,
            passed=True,
            word_count=2000,
            duplicate_score=0.80,
        )
        passed, reasons = scorer.check_gate(report)
        assert passed is False
        assert any("near-duplicate" in r for r in reasons)


# ===========================================================================
# VALID SITE IDS
# ===========================================================================


class TestValidSiteIds:

    def test_sixteen_site_ids(self):
        assert len(VALID_SITE_IDS) == 16

    def test_known_ids_present(self):
        for sid in ("witchcraft", "smarthome", "aiaction", "family"):
            assert sid in VALID_SITE_IDS


# ===========================================================================
# DEFAULT WEIGHTS
# ===========================================================================


class TestDefaultWeights:

    def test_seven_weights(self):
        assert len(DEFAULT_WEIGHTS) == 7

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_positive(self):
        for v in DEFAULT_WEIGHTS.values():
            assert v > 0


# ===========================================================================
# SYNC WRAPPER
# ===========================================================================


class TestSyncWrapper:

    def test_score_sync(self):
        scorer = ContentQualityScorer()
        report = scorer.score_sync(
            content="<h2>Heading</h2><p>A decent paragraph of text to test the scorer.</p>",
            title="Sync Test",
            site_id="witchcraft",
        )
        assert isinstance(report, QualityReport)
        assert report.overall_score >= 0.0
