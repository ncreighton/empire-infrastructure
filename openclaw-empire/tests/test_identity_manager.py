"""Test identity_manager — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.identity_manager import (
        IdentityManager,
        Persona,
        PersonaDemographics,
        PersonaPersonality,
        PlatformProfile,
        EmailIdentity,
        IdentityGroup,
        Gender,
        AgeRange,
        Platform,
        PersonaStatus,
        IdentityTier,
        AGE_RANGE_BOUNDS,
        PLATFORM_BIO_LIMITS,
        NICHE_INTERESTS,
        FIRST_NAMES_MALE,
        FIRST_NAMES_FEMALE,
        LAST_NAMES,
        US_CITIES,
        OCCUPATIONS,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="identity_manager not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def data_dir(tmp_path):
    """Isolated data directory for test runs."""
    d = tmp_path / "identities"
    d.mkdir()
    return d


@pytest.fixture
def manager(data_dir):
    """IdentityManager with isolated data dir."""
    return IdentityManager(data_dir=data_dir)


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_gender_values(self):
        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.NON_BINARY.value == "non_binary"
        assert Gender.UNSPECIFIED.value == "unspecified"

    def test_age_range_values(self):
        assert AgeRange.TEEN.value == "teen"
        assert AgeRange.YOUNG_ADULT.value == "young_adult"
        assert AgeRange.SENIOR.value == "senior"

    def test_age_range_bounds_defined(self):
        for ar in AgeRange:
            assert ar in AGE_RANGE_BOUNDS
            lo, hi = AGE_RANGE_BOUNDS[ar]
            assert lo <= hi

    def test_platform_values(self):
        assert Platform.INSTAGRAM.value == "instagram"
        assert Platform.TIKTOK.value == "tiktok"
        assert Platform.GMAIL.value == "gmail"

    def test_persona_status_values(self):
        assert PersonaStatus.ACTIVE.value == "active"
        assert PersonaStatus.WARMING.value == "warming"
        assert PersonaStatus.BURNED.value == "burned"

    def test_identity_tier_values(self):
        assert IdentityTier.DISPOSABLE.value == "disposable"
        assert IdentityTier.PRIMARY.value == "primary"

    def test_platform_bio_limits(self):
        assert Platform.INSTAGRAM in PLATFORM_BIO_LIMITS
        limits = PLATFORM_BIO_LIMITS[Platform.INSTAGRAM]
        assert "bio" in limits and limits["bio"] == 150


# ===================================================================
# Data Class Tests
# ===================================================================


class TestPersonaDemographics:
    def test_defaults(self):
        d = PersonaDemographics()
        assert d.first_name == ""
        assert d.gender == Gender.UNSPECIFIED
        assert d.age == 25
        assert d.country == "United States"

    def test_to_dict(self):
        d = PersonaDemographics(first_name="Jane", gender=Gender.FEMALE, age=30)
        result = d.to_dict()
        assert result["first_name"] == "Jane"
        assert result["gender"] == "female"

    def test_from_dict_roundtrip(self):
        d = PersonaDemographics(first_name="Alice", last_name="Smith", gender=Gender.FEMALE, age=28)
        as_dict = d.to_dict()
        restored = PersonaDemographics.from_dict(as_dict)
        assert restored.first_name == "Alice"
        assert restored.gender == Gender.FEMALE

    def test_from_dict_empty(self):
        d = PersonaDemographics.from_dict({})
        assert d.first_name == ""

    def test_from_dict_invalid_gender(self):
        d = PersonaDemographics.from_dict({"gender": "unknown_gender"})
        assert d.gender == Gender.UNSPECIFIED


class TestPersonaPersonality:
    def test_defaults(self):
        p = PersonaPersonality()
        assert p.interests == []
        assert p.emoji_usage == "moderate"

    def test_from_dict(self):
        p = PersonaPersonality.from_dict({
            "interests": ["tarot", "hiking"],
            "communication_style": "warm and friendly",
        })
        assert len(p.interests) == 2
        assert p.communication_style == "warm and friendly"


class TestPlatformProfile:
    def test_to_dict(self):
        pp = PlatformProfile(platform=Platform.INSTAGRAM, username="test_user", bio="Hi")
        d = pp.to_dict()
        assert d["platform"] == "instagram"
        assert d["username"] == "test_user"

    def test_from_dict(self):
        pp = PlatformProfile.from_dict({
            "platform": "tiktok",
            "username": "dancer99",
            "status": "active",
        })
        assert pp.platform == Platform.TIKTOK
        assert pp.status == PersonaStatus.ACTIVE


class TestEmailIdentity:
    def test_to_dict(self):
        ei = EmailIdentity(provider=Platform.GMAIL, address="test@gmail.com")
        d = ei.to_dict()
        assert d["provider"] == "gmail"
        assert d["address"] == "test@gmail.com"

    def test_from_dict(self):
        ei = EmailIdentity.from_dict({"provider": "outlook", "address": "x@outlook.com"})
        assert ei.provider == Platform.OUTLOOK


class TestPersona:
    def test_creation(self):
        p = Persona(id="abc123", name="Test Person")
        assert p.id == "abc123"
        assert p.tier == IdentityTier.STANDARD
        assert p.status == PersonaStatus.TEMPLATE

    def test_to_dict_roundtrip(self):
        p = Persona(
            id="p1", name="Jane Doe",
            demographics=PersonaDemographics(first_name="Jane", last_name="Doe", gender=Gender.FEMALE),
            personality=PersonaPersonality(interests=["tarot"]),
            tags=["witchcraft"],
        )
        d = p.to_dict()
        assert d["demographics"]["first_name"] == "Jane"
        assert "tarot" in d["personality"]["interests"]

        restored = Persona.from_dict(d)
        assert restored.name == "Jane Doe"
        assert restored.demographics.first_name == "Jane"

    def test_from_dict_empty(self):
        p = Persona.from_dict({})
        assert p.name == "Unknown"


class TestIdentityGroup:
    def test_creation(self):
        g = IdentityGroup(id="g1", name="Witchcraft Ring", persona_ids=["p1", "p2"])
        assert len(g.persona_ids) == 2

    def test_to_dict_roundtrip(self):
        g = IdentityGroup(id="g1", name="Test Group", purpose="Testing")
        d = g.to_dict()
        restored = IdentityGroup.from_dict(d)
        assert restored.name == "Test Group"


# ===================================================================
# IdentityManager Tests
# ===================================================================


class TestIdentityManagerInit:
    def test_init_creates_data_dir(self, tmp_path):
        d = tmp_path / "new_dir" / "identities"
        mgr = IdentityManager(data_dir=d)
        assert d.exists()
        assert len(mgr._personas) == 0

    def test_loads_empty_state(self, manager):
        assert len(manager._personas) == 0
        assert len(manager._groups) == 0


class TestDemographicsGeneration:
    def test_generates_valid_demographics(self, manager):
        d = manager._generate_demographics()
        assert d.first_name != ""
        assert d.last_name != ""
        assert d.first_name in FIRST_NAMES_MALE + FIRST_NAMES_FEMALE
        assert d.last_name in LAST_NAMES

    def test_gender_male_uses_male_names(self, manager):
        d = manager._generate_demographics(gender=Gender.MALE)
        assert d.first_name in FIRST_NAMES_MALE

    def test_gender_female_uses_female_names(self, manager):
        d = manager._generate_demographics(gender=Gender.FEMALE)
        assert d.first_name in FIRST_NAMES_FEMALE

    def test_age_within_range(self, manager):
        d = manager._generate_demographics(age_range=AgeRange.TEEN)
        lo, hi = AGE_RANGE_BOUNDS[AgeRange.TEEN]
        assert lo <= d.age <= hi

    def test_location_is_us_city(self, manager):
        d = manager._generate_demographics()
        city_names = [c["city"] for c in US_CITIES]
        assert d.city in city_names

    def test_occupation_is_valid(self, manager):
        d = manager._generate_demographics()
        assert d.occupation in OCCUPATIONS


class TestPersonalityGeneration:
    def test_generates_interests(self, manager):
        demo = PersonaDemographics(first_name="Test", age_range=AgeRange.ADULT, age=30)
        pers = manager._generate_personality(demo)
        assert len(pers.interests) > 0
        assert pers.communication_style != ""

    def test_niche_witchcraft_includes_relevant_interests(self, manager):
        demo = PersonaDemographics(first_name="Test", age_range=AgeRange.ADULT, age=30)
        pers = manager._generate_personality(demo, niche="witchcraft")
        niche_pool = NICHE_INTERESTS["witchcraft"]
        overlap = set(pers.interests) & set(niche_pool)
        assert len(overlap) > 0, "Expected witchcraft interests to appear"


class TestPersistence:
    def test_save_and_reload_personas(self, data_dir):
        mgr = IdentityManager(data_dir=data_dir)
        persona_data = {
            "id": "test-id",
            "name": "Test Persona",
            "demographics": PersonaDemographics(first_name="Test").to_dict(),
            "personality": PersonaPersonality(interests=["test"]).to_dict(),
            "status": PersonaStatus.ACTIVE.value,
            "tier": IdentityTier.STANDARD.value,
        }
        mgr._personas["test-id"] = persona_data
        mgr._save_personas()

        mgr2 = IdentityManager(data_dir=data_dir)
        assert "test-id" in mgr2._personas
        assert mgr2._personas["test-id"]["name"] == "Test Persona"

    def test_save_and_reload_groups(self, data_dir):
        mgr = IdentityManager(data_dir=data_dir)
        mgr._groups["g1"] = {"id": "g1", "name": "Test Group", "persona_ids": ["p1"]}
        mgr._save_groups()

        mgr2 = IdentityManager(data_dir=data_dir)
        assert "g1" in mgr2._groups


class TestHaikuIntegration:
    @pytest.mark.asyncio
    async def test_call_haiku_returns_empty_on_import_error(self, manager):
        with patch.dict("sys.modules", {"anthropic": None}):
            result = await manager._call_haiku("Generate a name")
            assert result == ""

    @pytest.mark.asyncio
    async def test_call_haiku_returns_text_on_success(self, manager):
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  Generated Name  ")]
        mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = await manager._call_haiku("Generate a name")
            assert result == "Generated Name"


# ===================================================================
# Niche Data Tests
# ===================================================================


class TestNicheData:
    def test_all_niches_have_interests(self):
        for niche in ["witchcraft", "smart home", "ai", "parenting", "mythology", "bullet journal"]:
            assert niche in NICHE_INTERESTS
            assert len(NICHE_INTERESTS[niche]) >= 4

    def test_us_cities_have_required_fields(self):
        for city in US_CITIES:
            assert "city" in city
            assert "state" in city
            assert "timezone" in city
