"""Tests for the Proactive Intelligence system."""

import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from moneyclaw.memory import Memory
from moneyclaw.services.luna.proactive import ProactiveIntelligence
from moneyclaw.services.luna.conv_memory import ConversationalMemory
from moneyclaw.services.luna.companion import CompanionEngine


def _tmp_memory():
    return Memory(db_path=Path(tempfile.mktemp(suffix=".db")))


def test_proactive_init():
    mem = _tmp_memory()
    pi = ProactiveIntelligence(mem)
    assert pi.memory is mem
    assert pi.conv_memory is not None
    assert pi.grimoire is not None


def test_generate_events_empty_user():
    mem = _tmp_memory()
    pi = ProactiveIntelligence(mem)
    events = pi.generate_events("u_new")
    # Should at least check moon — may or may not generate moon_alert depending on current phase
    assert isinstance(events, list)


def test_moon_alert_generation():
    mem = _tmp_memory()
    pi = ProactiveIntelligence(mem)
    from moneyclaw.services.luna.persona import get_moon_phase
    moon = get_moon_phase()

    events = pi.generate_events("u1")
    moon_events = [e for e in events if e["event_type"] == "moon_alert"]
    if moon.get("key") in ("new", "full"):
        assert len(moon_events) >= 1
    # If not new/full moon, no moon alert expected


def test_card_pattern_insight():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    # Log the same card multiple times
    for i in range(3):
        cm.log_card("u1", "The Tower", False, "daily_pull", "single")

    pi = ProactiveIntelligence(mem)
    events = pi.generate_events("u1")
    pattern_events = [e for e in events if e["event_type"] == "card_pattern_insight"]
    assert len(pattern_events) >= 1
    assert "Tower" in pattern_events[0]["payload"]["card"]


def test_followup_check():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    # Create a followup that's already due
    with mem._conn() as conn:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO luna_followups (user_id, promise_text, due_at, status) VALUES (?, ?, ?, 'pending')",
            ("u1", "Check on the interview", past)
        )

    pi = ProactiveIntelligence(mem)
    events = pi.generate_events("u1")
    followup_events = [e for e in events if e["event_type"] == "followup_check"]
    assert len(followup_events) >= 1
    assert "interview" in followup_events[0]["payload"]["promise"]


def test_get_due_events():
    mem = _tmp_memory()
    pi = ProactiveIntelligence(mem)

    # Schedule an event in the past (should be due)
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with mem._conn() as conn:
        conn.execute(
            "INSERT INTO proactive_events (user_id, event_type, payload_json, scheduled_for, status) VALUES (?, ?, ?, ?, 'pending')",
            ("u1", "test_event", '{"msg": "test"}', past)
        )

    due = pi.get_due_events()
    assert len(due) >= 1
    assert due[0]["event_type"] == "test_event"


def test_mark_sent():
    mem = _tmp_memory()
    pi = ProactiveIntelligence(mem)

    past = datetime.now(timezone.utc).isoformat()
    with mem._conn() as conn:
        conn.execute(
            "INSERT INTO proactive_events (user_id, event_type, payload_json, scheduled_for, status) VALUES (?, ?, ?, ?, 'pending')",
            ("u1", "test_event", '{}', past)
        )

    due = pi.get_due_events()
    assert len(due) >= 1
    pi.mark_sent(due[0]["id"])

    due_after = pi.get_due_events()
    sent_events = [e for e in due_after if e["id"] == due[0]["id"]]
    assert len(sent_events) == 0  # Should no longer be in pending


def test_no_duplicate_events():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    for i in range(3):
        cm.log_card("u1", "The Moon", False, "daily_pull", "single")

    pi = ProactiveIntelligence(mem)
    events1 = pi.generate_events("u1")
    events2 = pi.generate_events("u1")
    # Second call should not generate same card_pattern_insight again
    card_events_2 = [e for e in events2 if e["event_type"] == "card_pattern_insight"]
    assert len(card_events_2) == 0


def test_birthday_reading():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "My birthday is March 21st.")

    pi = ProactiveIntelligence(mem)
    events = pi.generate_events("u1")
    birthday_events = [e for e in events if e["event_type"] == "birthday_reading"]
    assert len(birthday_events) >= 1
