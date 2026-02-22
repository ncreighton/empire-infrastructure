"""Tests for grimoire.forge.practice_codex — SQLite Learning Engine."""

import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.forge.practice_codex import PracticeCodex
from grimoire.models import PracticeEntry, TarotReading, JourneyInsight


def _make_codex() -> PracticeCodex:
    """Create a fresh in-memory PracticeCodex for each test."""
    return PracticeCodex(db_path=":memory:")


def _make_entry(**kwargs) -> PracticeEntry:
    """Convenience helper to build a PracticeEntry with defaults."""
    defaults = dict(
        practice_type="spell",
        title="Protection Jar",
        intention="ward my home",
        moon_phase="full_moon",
        correspondences_used=["rosemary", "black tourmaline", "salt"],
        effectiveness_rating=4,
    )
    defaults.update(kwargs)
    return PracticeEntry(**defaults)


def test_log_practice():
    """Logs an entry and returns an id > 0."""
    codex = _make_codex()
    entry = _make_entry()
    practice_id = codex.log_practice(entry)
    assert isinstance(practice_id, int)
    assert practice_id > 0


def test_log_moon_journal():
    """Logs a moon journal entry and returns an id > 0."""
    codex = _make_codex()
    entry_id = codex.log_moon_journal(
        moon_phase="full_moon",
        entry="The full moon felt incredibly powerful tonight.",
        mood="energized",
        dreams="Dreamed of a silver river.",
        energy_level=8,
    )
    assert isinstance(entry_id, int)
    assert entry_id > 0


def test_log_tarot_reading():
    """Logs a tarot reading and returns an id > 0."""
    codex = _make_codex()
    reading = TarotReading(
        spread_name="Three Card Spread",
        question="What guidance do I need today?",
        cards=[
            {"position": "Past", "card": "The Fool", "orientation": "upright"},
            {"position": "Present", "card": "The Star", "orientation": "upright"},
            {"position": "Future", "card": "The World", "orientation": "reversed"},
        ],
        interpretation="A journey from innocence to hope to completion.",
    )
    reading_id = codex.log_tarot_reading(reading)
    assert isinstance(reading_id, int)
    assert reading_id > 0


def test_log_intention():
    """Logs an intention with pending status and returns an id > 0."""
    codex = _make_codex()
    intention_id = codex.log_intention(
        intention="Manifest abundance in my career",
        category="prosperity",
        method="candle spell",
        correspondences=["cinnamon", "citrine", "green"],
        moon_phase="waxing_gibbous",
    )
    assert isinstance(intention_id, int)
    assert intention_id > 0


def test_update_intention_outcome():
    """Updates the outcome to 'manifested' without error."""
    codex = _make_codex()
    intention_id = codex.log_intention(
        intention="Find inner peace",
        category="peace",
        method="meditation",
        correspondences=["lavender", "amethyst"],
    )
    # Should not raise
    codex.update_intention_outcome(intention_id, "manifested", notes="Felt very calm")


def test_get_stats_empty():
    """Empty db returns all zeros."""
    codex = _make_codex()
    stats = codex.get_stats()
    assert stats["total_sessions"] == 0
    assert stats["streak"] == 0
    assert stats["total_tarot_readings"] == 0
    assert stats["total_moon_journal_entries"] == 0
    assert stats["total_intentions"] == 0
    assert stats["manifested_intentions"] == 0


def test_get_stats_after_logging():
    """After logging a session, total_sessions > 0."""
    codex = _make_codex()
    codex.log_practice(_make_entry())
    stats = codex.get_stats()
    assert stats["total_sessions"] > 0


def test_practice_streak():
    """Log 2 consecutive days and verify streak = 2."""
    codex = _make_codex()
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    entry_today = _make_entry(date=today.isoformat())
    entry_yesterday = _make_entry(
        title="Yesterday's spell",
        date=yesterday.isoformat(),
    )
    codex.log_practice(entry_yesterday)
    codex.log_practice(entry_today)

    streak = codex.get_practice_streak()
    assert streak == 2, f"Expected streak of 2, got {streak}"


def test_favorite_correspondences():
    """After logging with correspondences, get_favorite_correspondences returns them."""
    codex = _make_codex()
    codex.log_practice(_make_entry(
        correspondences_used=["rosemary", "sage", "rosemary"],
    ))
    codex.log_practice(_make_entry(
        correspondences_used=["rosemary", "lavender"],
    ))
    favs = codex.get_favorite_correspondences(limit=5)
    assert isinstance(favs, list)
    assert len(favs) > 0
    # Rosemary should appear since it was used most
    names = [f["name"] for f in favs]
    assert "rosemary" in names


def test_personalized_recommendations():
    """Returns a non-empty list of recommendation strings."""
    codex = _make_codex()
    # With no data, should still return at least one recommendation (welcome message)
    recs = codex.get_personalized_recommendations()
    assert isinstance(recs, list)
    assert len(recs) > 0
    for rec in recs:
        assert isinstance(rec, str)


def test_growth_summary():
    """Returns a JourneyInsight dataclass."""
    codex = _make_codex()
    codex.log_practice(_make_entry())
    summary = codex.get_growth_summary()
    assert isinstance(summary, JourneyInsight)
    assert summary.total_sessions >= 1
    assert isinstance(summary.recommendations, list)
    assert isinstance(summary.next_sabbat, str)
