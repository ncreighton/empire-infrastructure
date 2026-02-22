"""Tests for CodexAdvisor — personalization bridge from PracticeCodex."""

import pytest
from grimoire.forge.practice_codex import PracticeCodex
from grimoire.forge.codex_advisor import CodexAdvisor
from grimoire.models import PracticeEntry


@pytest.fixture
def codex():
    """Create an in-memory PracticeCodex for tests."""
    return PracticeCodex(db_path=":memory:")


@pytest.fixture
def advisor(codex):
    """Create a CodexAdvisor backed by the in-memory codex."""
    return CodexAdvisor(codex)


def _log_session(codex, practice_type="spell", title="Test", intention="protection",
                 correspondences=None, effectiveness=4, moon_phase="Full Moon"):
    """Helper to log a practice session."""
    entry = PracticeEntry(
        practice_type=practice_type,
        title=title,
        intention=intention,
        correspondences_used=correspondences or ["rosemary", "black tourmaline"],
        effectiveness_rating=effectiveness,
        moon_phase=moon_phase,
    )
    return codex.log_practice(entry)


# ── Empty DB fallback ────────────────────────────────────────────────────


class TestEmptyDB:
    def test_preferred_herbs_empty(self, advisor):
        assert advisor.get_preferred_herbs("protection") == []

    def test_preferred_crystals_empty(self, advisor):
        assert advisor.get_preferred_crystals("love") == []

    def test_auto_difficulty_defaults_beginner(self, advisor):
        assert advisor.get_auto_difficulty() == "beginner"

    def test_best_moon_phase_returns_none(self, advisor):
        assert advisor.get_best_moon_phase() is None

    def test_personalization_context_empty_db(self, advisor):
        ctx = advisor.get_personalization_context("protection")
        assert ctx["session_count"] == 0
        assert ctx["streak"] == 0
        assert ctx["auto_difficulty"] == "beginner"
        assert ctx["preferred_herbs"] == []
        assert ctx["preferred_crystals"] == []
        # Discovery candidates may return items from the knowledge base
        # since a new user hasn't used anything yet
        assert isinstance(ctx["discovery_herb"], str)
        assert isinstance(ctx["discovery_crystal"], str)
        assert ctx["best_moon_phase"] is None
        assert ctx["top_method"] == ""


# ── With data ────────────────────────────────────────────────────────────


class TestWithData:
    def test_preferred_herbs_returns_logged_herbs(self, advisor, codex):
        _log_session(codex, correspondences=["rosemary", "sage"], effectiveness=5)
        _log_session(codex, correspondences=["rosemary", "lavender"], effectiveness=4)
        herbs = advisor.get_preferred_herbs("protection")
        assert "rosemary" in herbs

    def test_preferred_crystals_returns_logged(self, advisor, codex):
        _log_session(codex, correspondences=["black tourmaline", "obsidian"], effectiveness=5)
        crystals = advisor.get_preferred_crystals("protection")
        assert "black tourmaline" in crystals or "obsidian" in crystals

    def test_auto_difficulty_intermediate(self, advisor, codex):
        for i in range(15):
            _log_session(codex, title=f"Session {i}")
        assert advisor.get_auto_difficulty() == "intermediate"

    def test_auto_difficulty_advanced(self, advisor, codex):
        for i in range(55):
            _log_session(codex, title=f"Session {i}")
        assert advisor.get_auto_difficulty() == "advanced"

    def test_best_moon_phase_with_data(self, advisor, codex):
        # Log 3+ sessions with effectiveness during Full Moon
        for i in range(4):
            _log_session(codex, effectiveness=5, moon_phase="Full Moon", title=f"FM {i}")
        # Log 1 session during New Moon
        _log_session(codex, effectiveness=3, moon_phase="New Moon", title="NM 1")
        result = advisor.get_best_moon_phase()
        assert result == "Full Moon"

    def test_personalization_context_populated(self, advisor, codex):
        for i in range(5):
            _log_session(
                codex,
                correspondences=["rosemary", "citrine"],
                effectiveness=4,
                title=f"Session {i}",
            )
        ctx = advisor.get_personalization_context("protection")
        assert ctx["session_count"] == 5
        assert ctx["auto_difficulty"] == "beginner"
        assert len(ctx["preferred_herbs"]) > 0 or len(ctx["preferred_crystals"]) > 0

    def test_personalization_context_has_all_keys(self, advisor, codex):
        _log_session(codex)
        ctx = advisor.get_personalization_context()
        expected_keys = {
            "session_count", "streak", "auto_difficulty",
            "preferred_herbs", "preferred_crystals",
            "discovery_herb", "discovery_crystal",
            "best_moon_phase", "top_method", "favorite_correspondences",
        }
        assert expected_keys.issubset(set(ctx.keys()))


# ── Discovery candidates ────────────────────────────────────────────────


class TestDiscovery:
    def test_discovery_returns_unused_herbs(self, advisor, codex):
        # Log a session using rosemary
        _log_session(codex, correspondences=["rosemary"])
        # Discovery should NOT include rosemary
        candidates = advisor.get_discovery_candidates("protection", "herb", 5)
        assert "rosemary" not in candidates

    def test_discovery_empty_when_no_kb(self, advisor):
        # Unknown category returns empty
        candidates = advisor.get_discovery_candidates("love", "potion", 3)
        assert candidates == []
