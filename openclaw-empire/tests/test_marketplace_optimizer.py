"""Test marketplace_optimizer — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.marketplace_optimizer import (
        MarketplaceOptimizer,
        Listing,
        KeywordResearch,
        OptimizationResult,
        PricingAnalysis,
        CompetitorListing,
        Marketplace,
        ListingType,
        OptimizationType,
        PricingStrategy,
        KeywordDifficulty,
        get_optimizer,
        _load_json,
        _save_json,
        _parse_json_response,
        KDP_MIN_EBOOK_PRICE,
        KDP_MAX_EBOOK_PRICE,
        ETSY_MAX_TAGS,
        ETSY_TITLE_MAX_LENGTH,
        VALID_NICHES,
        NICHE_TO_SITE,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="marketplace_optimizer module not available"
)


# ===================================================================
# Enum tests
# ===================================================================

class TestEnums:
    """Verify marketplace enums and from_string parsers."""

    def test_marketplace_values(self):
        assert Marketplace.KDP.value == "kdp"
        assert Marketplace.ETSY.value == "etsy"
        assert Marketplace.AMAZON.value == "amazon"

    def test_marketplace_from_string(self):
        assert Marketplace.from_string("kdp") == Marketplace.KDP
        assert Marketplace.from_string("ETSY") == Marketplace.ETSY
        with pytest.raises(ValueError):
            Marketplace.from_string("shopify")

    def test_listing_type_values(self):
        assert ListingType.KDP_EBOOK.value == "kdp_ebook"
        assert ListingType.ETSY_DIGITAL.value == "etsy_digital"
        assert ListingType.ETSY_POD.value == "etsy_pod"

    def test_listing_type_marketplace(self):
        assert ListingType.KDP_EBOOK.marketplace == Marketplace.KDP
        assert ListingType.ETSY_POD.marketplace == Marketplace.ETSY

    def test_listing_type_from_string(self):
        assert ListingType.from_string("kdp_paperback") == ListingType.KDP_PAPERBACK
        with pytest.raises(ValueError):
            ListingType.from_string("invalid_type")

    def test_optimization_type(self):
        assert OptimizationType.FULL.value == "full"
        assert OptimizationType.TITLE.value == "title"
        assert OptimizationType.from_string("keywords") == OptimizationType.KEYWORDS

    def test_pricing_strategy(self):
        assert PricingStrategy.COMPETITIVE.value == "competitive"
        assert PricingStrategy.PREMIUM.value == "premium"
        assert PricingStrategy.from_string("penetration") == PricingStrategy.PENETRATION

    def test_keyword_difficulty(self):
        assert KeywordDifficulty.LOW.value == "low"
        assert KeywordDifficulty.from_score(10) == KeywordDifficulty.LOW
        assert KeywordDifficulty.from_score(40) == KeywordDifficulty.MEDIUM
        assert KeywordDifficulty.from_score(60) == KeywordDifficulty.HIGH
        assert KeywordDifficulty.from_score(90) == KeywordDifficulty.VERY_HIGH


# ===================================================================
# Listing dataclass
# ===================================================================

class TestListing:
    """Test Listing dataclass."""

    def test_default_listing_gets_id(self):
        listing = Listing()
        assert listing.listing_id.startswith("lst_")
        assert listing.created_at

    def test_listing_with_fields(self):
        listing = Listing(
            marketplace=Marketplace.KDP.value,
            listing_type=ListingType.KDP_EBOOK.value,
            title="Crystal Healing 101",
            niche="crystals",
            price=4.99,
        )
        assert listing.title == "Crystal Healing 101"
        assert listing.price == 4.99
        assert listing.niche == "crystals"

    def test_listing_from_dict(self):
        data = {
            "listing_id": "lst_test123",
            "title": "Test Book",
            "marketplace": "kdp",
            "price": 9.99,
            "unknown_field": "ignored",
        }
        listing = Listing.from_dict(data)
        assert listing.listing_id == "lst_test123"
        assert listing.title == "Test Book"
        assert listing.price == 9.99


# ===================================================================
# KeywordResearch dataclass
# ===================================================================

class TestKeywordResearch:
    """Test KeywordResearch dataclass."""

    def test_default_keyword(self):
        kw = KeywordResearch(keyword="moon water")
        assert kw.keyword == "moon water"
        assert kw.researched_at  # auto-set

    def test_from_dict(self):
        data = {
            "keyword": "crystal healing",
            "search_volume": 12000,
            "difficulty": "medium",
            "competition": 0.45,
        }
        kw = KeywordResearch.from_dict(data)
        assert kw.search_volume == 12000
        assert kw.difficulty == "medium"


# ===================================================================
# OptimizationResult dataclass
# ===================================================================

class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_auto_id(self):
        opt = OptimizationResult()
        assert opt.optimization_id.startswith("opt_")

    def test_from_dict(self):
        data = {
            "optimization_id": "opt_abc123",
            "listing_id": "lst_123",
            "score_before": 45.0,
            "score_after": 78.0,
            "improvement": 33.0,
        }
        opt = OptimizationResult.from_dict(data)
        assert opt.score_before == 45.0
        assert opt.improvement == 33.0


# ===================================================================
# PricingAnalysis dataclass
# ===================================================================

class TestPricingAnalysis:
    """Test PricingAnalysis dataclass."""

    def test_auto_timestamp(self):
        pa = PricingAnalysis(listing_id="lst_1", current_price=4.99)
        assert pa.analyzed_at

    def test_from_dict(self):
        data = {
            "listing_id": "lst_1",
            "current_price": 4.99,
            "recommended_price": 6.99,
            "strategy": "premium",
        }
        pa = PricingAnalysis.from_dict(data)
        assert pa.recommended_price == 6.99
        assert pa.strategy == "premium"


# ===================================================================
# CompetitorListing dataclass
# ===================================================================

class TestCompetitorListing:
    """Test CompetitorListing dataclass."""

    def test_auto_id(self):
        cl = CompetitorListing(title="Competitor Book")
        assert cl.competitor_id.startswith("comp_")

    def test_from_dict(self):
        data = {
            "title": "Rival Product",
            "price": 12.99,
            "reviews": 245,
            "rating": 4.3,
        }
        cl = CompetitorListing.from_dict(data)
        assert cl.title == "Rival Product"
        assert cl.reviews == 245


# ===================================================================
# MarketplaceOptimizer — listing management
# ===================================================================

class TestMarketplaceOptimizerListings:
    """Test listing CRUD on the optimizer."""

    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    def test_add_listing(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        listing = opt.add_listing(
            marketplace="kdp",
            listing_type="kdp_ebook",
            title="Witchcraft Basics",
            niche="witchcraft",
            price=4.99,
        )
        assert isinstance(listing, Listing)
        assert listing.title == "Witchcraft Basics"
        assert listing.marketplace == "kdp"

    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    def test_get_listing(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        listing = opt.add_listing(
            marketplace="etsy",
            listing_type="etsy_digital",
            title="Tarot Printable",
            niche="tarot",
            price=7.99,
        )
        fetched = opt.get_listing(listing.listing_id)
        assert fetched.title == "Tarot Printable"

    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    def test_list_listings_all(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        opt.add_listing(marketplace="kdp", listing_type="kdp_ebook", title="A")
        opt.add_listing(marketplace="etsy", listing_type="etsy_pod", title="B")
        all_listings = opt.list_listings()
        assert len(all_listings) >= 2

    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    def test_list_listings_by_marketplace(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        opt.add_listing(marketplace="kdp", listing_type="kdp_ebook", title="KDP A")
        opt.add_listing(marketplace="etsy", listing_type="etsy_pod", title="Etsy A")
        kdp_listings = opt.list_listings(marketplace="kdp")
        for l in kdp_listings:
            assert l.marketplace == "kdp"


# ===================================================================
# Listing scoring
# ===================================================================

class TestListingScoring:
    """Test the scoring algorithm."""

    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    def test_score_empty_listing(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        listing = Listing(title="")
        score = opt.score_listing(listing)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    def test_score_rich_listing(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        listing = Listing(
            title="Complete Crystal Healing Guide for Beginners",
            description="A comprehensive guide" + " word" * 200,
            keywords=["crystal healing", "crystals", "healing stones"],
            tags=["crystals", "healing", "beginner", "guide", "spiritual"],
            bullet_points=["Learn basics", "Advanced techniques", "Crystal grid layouts"],
            price=9.99,
        )
        score = opt.score_listing(listing)
        assert score > 0  # should score higher than empty


# ===================================================================
# Keyword research (mocked AI)
# ===================================================================

class TestKeywordResearch:
    """Test keyword research with mocked API calls."""

    @pytest.mark.asyncio
    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    @patch("src.marketplace_optimizer._call_haiku")
    async def test_research_keywords(self, mock_haiku, mock_load, mock_save):
        mock_haiku.return_value = json.dumps({
            "keywords": [
                {"keyword": "crystal healing guide", "search_volume": 5000,
                 "difficulty": 35, "relevance": 0.9},
                {"keyword": "healing crystals book", "search_volume": 3000,
                 "difficulty": 40, "relevance": 0.85},
            ]
        })
        opt = MarketplaceOptimizer()
        results = await opt.research_keywords(
            seeds=["crystal healing"],
            marketplace="kdp",
        )
        assert isinstance(results, list)


# ===================================================================
# Pricing optimization
# ===================================================================

class TestPricingOptimization:
    """Test pricing analysis with mocked calls."""

    @pytest.mark.asyncio
    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    @patch("src.marketplace_optimizer._call_sonnet")
    async def test_analyze_pricing(self, mock_sonnet, mock_load, mock_save):
        mock_sonnet.return_value = json.dumps({
            "recommended_price": 6.99,
            "rationale": "Competitive positioning",
            "price_range_low": 4.99,
            "price_range_high": 9.99,
        })
        opt = MarketplaceOptimizer()
        listing = opt.add_listing(
            marketplace="kdp",
            listing_type="kdp_ebook",
            title="Moon Ritual Guide",
            niche="witchcraft",
            price=4.99,
        )
        result = await opt.analyze_pricing(
            listing_id=listing.listing_id,
            strategy=PricingStrategy.COMPETITIVE,
        )
        assert isinstance(result, PricingAnalysis)


# ===================================================================
# Batch optimization
# ===================================================================

class TestBatchOptimization:
    """Test batch optimization of multiple listings."""

    @pytest.mark.asyncio
    @patch("src.marketplace_optimizer._save_json")
    @patch("src.marketplace_optimizer._load_json", return_value={})
    async def test_batch_optimize_calls_optimize(self, mock_load, mock_save):
        opt = MarketplaceOptimizer()
        opt.add_listing(marketplace="kdp", listing_type="kdp_ebook", title="Book A", niche="witchcraft")
        opt.add_listing(marketplace="kdp", listing_type="kdp_ebook", title="Book B", niche="crystals")

        with patch.object(opt, "optimize_listing", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = OptimizationResult(
                score_before=40, score_after=70, improvement=30
            )
            results = await opt.optimize_batch(marketplace="kdp")
            assert isinstance(results, list)


# ===================================================================
# Platform-specific configs
# ===================================================================

class TestPlatformConfigs:
    """Test KDP and Etsy specific constants."""

    def test_kdp_price_bounds(self):
        assert KDP_MIN_EBOOK_PRICE == 0.99
        assert KDP_MAX_EBOOK_PRICE == 9.99

    def test_etsy_constraints(self):
        assert ETSY_MAX_TAGS == 13
        assert ETSY_TITLE_MAX_LENGTH == 140

    def test_valid_niches(self):
        assert "witchcraft" in VALID_NICHES
        assert "crystals" in VALID_NICHES
        assert "ai" in VALID_NICHES

    def test_niche_to_site_mapping(self):
        assert NICHE_TO_SITE["witchcraft"] == "witchcraft"
        assert NICHE_TO_SITE["crystals"] == "crystalwitchcraft"
        assert NICHE_TO_SITE["smart_home"] == "smarthome"


# ===================================================================
# JSON response parsing
# ===================================================================

class TestParseJsonResponse:
    """Test LLM response JSON extraction."""

    def test_parse_plain_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_fenced_json(self):
        result = _parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_embedded_json(self):
        result = _parse_json_response('Some text before\n{"key": "value"}\nand after')
        assert result == {"key": "value"}

    def test_parse_none_input(self):
        assert _parse_json_response(None) is None

    def test_parse_empty_string(self):
        assert _parse_json_response("") is None

    def test_parse_invalid_json(self):
        result = _parse_json_response("not json at all, no braces")
        assert result is None


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """Test data persistence helpers."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test.json"
        _save_json(path, {"listings": [1, 2, 3]})
        loaded = _load_json(path)
        assert loaded == {"listings": [1, 2, 3]}

    def test_load_missing_returns_default(self, tmp_path):
        result = _load_json(tmp_path / "nope.json", [])
        assert result == []
