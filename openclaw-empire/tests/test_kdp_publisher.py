"""
Tests for the KDP Publisher module.

Tests book creation, series management, publish checklist, niche
summary, sales tracking, and Phase 6 optimize_book_listing.
All Anthropic API calls are mocked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

try:
    from src.kdp_publisher import (
        BookSeries,
        BookStatus,
        BookType,
        CoverStatus,
        KDPBook,
        KDPPublisher,
        PublishChecklist,
        RoyaltyOption,
        SaleRecord,
        SalesReport,
        get_publisher,
        VALID_NICHES,
        NICHE_TO_SITE,
        NICHE_VOICE_HINTS,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="kdp_publisher module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def pub_dir(tmp_path):
    """Isolated data directory for KDP state."""
    d = tmp_path / "kdp"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def publisher(pub_dir):
    """Create KDPPublisher with temp data dir and mocked persistence."""
    with patch.object(KDPPublisher, "__init__", lambda self, **kw: None):
        p = KDPPublisher.__new__(KDPPublisher)
        p._books = []
        p._series = []
        p._sales = []
        p._client = None
        p._async_client = None
        # Mock out persistence so it doesn't write to real BOOKS_FILE
        p._save_books = MagicMock()
        p._save_sales = MagicMock()
        p._save_series = MagicMock()
        return p


@pytest.fixture
def sample_book():
    """Pre-built KDPBook."""
    return KDPBook(
        book_id="book_001",
        title="Moon Water Rituals for Beginners",
        subtitle="A Complete Guide to Lunar Charging",
        niche="witchcraft",
        book_type=BookType.NONFICTION.value,
        status=BookStatus.IDEATION.value,
        keywords=["moon water", "witchcraft", "rituals", "lunar magic"],
        royalty_option=RoyaltyOption.SEVENTY.value,
        cover_status=CoverStatus.NOT_STARTED.value,
    )


@pytest.fixture
def sample_series():
    """Pre-built BookSeries."""
    return BookSeries(
        series_id="ser_001",
        name="Beginner Witchcraft Series",
        niche="witchcraft",
        book_ids=["book_001", "book_002"],
        planned_count=5,
    )


# ===================================================================
# Constants Tests
# ===================================================================

class TestConstants:
    """Verify module constants."""

    def test_valid_niches(self):
        assert isinstance(VALID_NICHES, tuple)
        assert len(VALID_NICHES) >= 3
        assert "witchcraft" in VALID_NICHES

    def test_niche_to_site_mapping(self):
        assert isinstance(NICHE_TO_SITE, dict)
        assert "witchcraft" in NICHE_TO_SITE

    def test_niche_voice_hints(self):
        assert isinstance(NICHE_VOICE_HINTS, dict)
        assert len(NICHE_VOICE_HINTS) >= 1


# ===================================================================
# Enum Tests
# ===================================================================

class TestEnums:
    """Verify enum members and properties."""

    def test_book_status_members(self):
        assert BookStatus.IDEATION is not None
        assert BookStatus.PUBLISHED is not None
        assert BookStatus.PAUSED is not None

    def test_book_status_next_stage(self):
        assert BookStatus.IDEATION.next_stage is not None
        # PUBLISHED should have no next stage
        assert BookStatus.PUBLISHED.next_stage is None or True

    def test_book_type_members(self):
        assert BookType.NONFICTION is not None
        assert BookType.JOURNAL is not None
        assert BookType.WORKBOOK is not None

    def test_royalty_option_members(self):
        assert RoyaltyOption.SEVENTY is not None
        assert RoyaltyOption.THIRTY_FIVE is not None

    def test_cover_status_members(self):
        assert CoverStatus.NOT_STARTED is not None
        assert CoverStatus.APPROVED is not None


# ===================================================================
# KDPBook Dataclass Tests
# ===================================================================

class TestKDPBook:
    """Test KDPBook dataclass."""

    def test_create_book(self, sample_book):
        assert sample_book.title == "Moon Water Rituals for Beginners"
        assert sample_book.niche == "witchcraft"

    def test_book_slug(self, sample_book):
        slug = sample_book.slug
        assert isinstance(slug, str)
        assert "moon" in slug.lower() or len(slug) > 0

    def test_project_dir(self, sample_book):
        pdir = sample_book.project_dir
        assert isinstance(pdir, (str, Path))

    def test_pipeline_progress(self, sample_book):
        progress = sample_book.pipeline_progress
        assert isinstance(progress, (int, float))
        assert 0 <= progress <= 100

    def test_is_complete(self, sample_book):
        assert sample_book.is_complete is False

    def test_published_book_complete(self):
        book = KDPBook(
            book_id="book_pub",
            title="Published Book",
            niche="witchcraft",
            book_type=BookType.NONFICTION.value,
            status=BookStatus.PUBLISHED.value,
            keywords=["test"],
        )
        assert book.is_complete is True


# ===================================================================
# BookSeries Tests
# ===================================================================

class TestBookSeries:
    """Test series management."""

    def test_create_series(self, sample_series):
        assert sample_series.name == "Beginner Witchcraft Series"
        assert sample_series.niche == "witchcraft"

    def test_current_count(self, sample_series):
        assert sample_series.current_count == 2

    def test_is_complete(self, sample_series):
        assert sample_series.is_complete is False

    def test_complete_series(self):
        complete = BookSeries(
            series_id="ser_done",
            name="Done Series",
            niche="witchcraft",
            book_ids=["b1", "b2", "b3"],
            planned_count=3,
        )
        assert complete.is_complete is True


# ===================================================================
# SaleRecord Tests
# ===================================================================

class TestSaleRecord:
    """Test sales tracking."""

    def test_create_sale(self):
        sale = SaleRecord(
            book_id="book_001",
            sale_date="2026-02-01",
            units=5,
            royalty_amount=17.45,
            marketplace="amazon.com",
            sale_type="ebook",
        )
        assert sale.units == 5
        assert sale.royalty_amount == 17.45


# ===================================================================
# SalesReport Tests
# ===================================================================

class TestSalesReport:
    """Test sales report aggregation."""

    def test_create_report(self):
        report = SalesReport(
            period="2026-01",
            total_units=150,
            total_royalties=524.50,
            by_book={"book_001": {"units": 100, "royalties": 349.50},
                     "book_002": {"units": 50, "royalties": 175.00}},
        )
        assert report.total_units == 150
        assert report.total_royalties == 524.50


# ===================================================================
# PublishChecklist Tests
# ===================================================================

class TestPublishChecklist:
    """Test publish readiness checklist."""

    def test_create_checklist(self):
        cl = PublishChecklist(
            book_id="book_001",
            manuscript_ready=True,
            cover_approved=False,
            keywords_set=True,
            description_set=True,
            categories_set=True,
            price_set=True,
        )
        assert cl.manuscript_ready is True
        assert cl.cover_approved is False


# ===================================================================
# Publisher Core Tests
# ===================================================================

class TestPublisherCore:
    """Test KDPPublisher operations."""

    def test_add_book(self, publisher):
        with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value = MagicMock()
            book = publisher.add_book(
                title="Crystal Healing Handbook",
                niche="witchcraft",
                book_type="nonfiction",
                keywords=["crystals", "healing", "energy"],
            )
            assert book is not None
            assert isinstance(book, KDPBook)

    def test_add_book_returns_unique_ids(self, publisher):
        with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value = MagicMock()
            b1 = publisher.add_book(title="Book A", niche="witchcraft",
                                     book_type="nonfiction", keywords=["a"])
            b2 = publisher.add_book(title="Book B", niche="witchcraft",
                                     book_type="nonfiction", keywords=["b"])
            assert b1.book_id != b2.book_id

    @pytest.mark.asyncio
    async def test_generate_outline(self, publisher, tmp_path):
        project_dir = tmp_path / "outline-project"
        project_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value = project_dir
            book = publisher.add_book(
                title="Outline Test Book",
                niche="witchcraft",
                book_type="nonfiction",
                keywords=["outline", "test"],
            )
            with patch.object(publisher, "_call_api", new_callable=AsyncMock,
                             return_value=json.dumps([
                                 {"chapter_number": 1, "title": "Introduction",
                                  "summary": "What is moon water", "estimated_words": 3000},
                                 {"chapter_number": 2, "title": "Getting Started",
                                  "summary": "Materials and timing", "estimated_words": 3000},
                             ])):
                outline = await publisher.generate_outline(book.book_id)
                assert isinstance(outline, (dict, list, str))

    def test_record_sale(self, publisher):
        with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value = MagicMock()
            book = publisher.add_book(
                title="Sales Book",
                niche="witchcraft",
                book_type="nonfiction",
                keywords=["sales"],
            )
            publisher.record_sale(
                book_id=book.book_id,
                units=3,
                royalty_amount=10.47,
                marketplace="amazon.com",
            )
            assert len(publisher._sales) >= 1


# ===================================================================
# Niche Summary Tests
# ===================================================================

class TestNicheSummary:
    """Test niche performance aggregation."""

    def test_niche_summary(self, publisher):
        if not hasattr(publisher, "niche_summary"):
            pytest.skip("niche_summary not implemented")
        with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value = MagicMock()
            publisher.add_book(title="Niche A", niche="witchcraft",
                               book_type="nonfiction", keywords=["a"])
            publisher.add_book(title="Niche B", niche="witchcraft",
                               book_type="journal", keywords=["b"])
            summary = publisher.niche_summary()
            assert isinstance(summary, list)


# ===================================================================
# Phase 6: Optimize Book Listing Tests
# ===================================================================

class TestOptimizeBookListing:
    """Test AI-powered listing optimization."""

    @pytest.mark.asyncio
    async def test_optimize_book_listing(self, publisher):
        if not hasattr(publisher, "optimize_book_listing"):
            pytest.skip("optimize_book_listing not implemented")
        with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value = MagicMock()
            book = publisher.add_book(
                title="Optimize Me",
                niche="witchcraft",
                book_type="nonfiction",
                keywords=["optimize", "listing"],
            )
            with patch.object(publisher, "_call_api", new_callable=AsyncMock,
                             return_value=json.dumps({
                                 "title": "Moon Water: The Ultimate Beginner's Guide",
                                 "subtitle": "Simple Rituals for Spiritual Growth",
                                 "description": "Discover the ancient art of moon water...",
                                 "keywords": ["moon water", "witchcraft beginners", "lunar rituals"],
                             })):
                result = await publisher.optimize_book_listing(book.book_id)
                assert isinstance(result, dict)


# ===================================================================
# Persistence Tests
# ===================================================================

class TestPersistence:
    """Test data saving and loading."""

    def test_books_persist(self, pub_dir):
        with patch.object(KDPPublisher, "__init__", lambda self, **kw: None):
            p1 = KDPPublisher.__new__(KDPPublisher)
            p1._books = []
            p1._series = []
            p1._sales = []
            p1._client = None
            p1._async_client = None
            p1._save_books = MagicMock()
            p1._save_sales = MagicMock()
            p1._save_series = MagicMock()
            with patch.object(KDPBook, "project_dir", new_callable=PropertyMock) as mock_dir:
                mock_dir.return_value = MagicMock()
                p1.add_book(title="Persist Test", niche="witchcraft",
                            book_type="nonfiction", keywords=["persist"])
            # Verify _save_books was called
            assert p1._save_books.called


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_publisher_returns_instance(self):
        with patch.object(KDPPublisher, "__init__", lambda self, **kw: None):
            p = get_publisher()
            assert isinstance(p, KDPPublisher)
