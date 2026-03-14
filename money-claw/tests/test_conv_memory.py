"""Tests for the Conversational Memory system."""

import tempfile
from pathlib import Path
from moneyclaw.memory import Memory
from moneyclaw.services.luna.conv_memory import ConversationalMemory


def _tmp_memory():
    return Memory(db_path=Path(tempfile.mktemp(suffix=".db")))


def test_extract_person():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "My husband John is really stressed about work.")
    profile = cm.get_user_profile("u1")
    people = profile["entities"].get("person", [])
    assert len(people) >= 1
    assert any(p["entity_name"] == "John" for p in people)


def test_extract_pet():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "I have a cat named Luna who sleeps on my altar.")
    profile = cm.get_user_profile("u1")
    pets = profile["entities"].get("pet", [])
    assert len(pets) >= 1
    assert any(p["entity_name"] == "Luna" for p in pets)


def test_extract_job():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "I'm a nurse at a children's hospital.")
    profile = cm.get_user_profile("u1")
    jobs = profile["entities"].get("job", [])
    assert len(jobs) >= 1


def test_extract_goal():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "I want to start a small business selling crystals.")
    profile = cm.get_user_profile("u1")
    goals = profile["entities"].get("goal", [])
    assert len(goals) >= 1


def test_extract_location():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "I live in Portland, Oregon.")
    profile = cm.get_user_profile("u1")
    locations = profile["entities"].get("location", [])
    assert len(locations) >= 1


def test_extract_date_event():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "My birthday is March 21st.")
    profile = cm.get_user_profile("u1")
    dates = profile["entities"].get("date_event", [])
    assert len(dates) >= 1


def test_mention_count_increments():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "My husband John is worried.")
    cm.extract_and_store("u1", "My husband John asked me to pull a card.")
    profile = cm.get_user_profile("u1")
    people = profile["entities"].get("person", [])
    johns = [p for p in people if p["entity_name"] == "John"]
    assert len(johns) == 1
    assert johns[0]["mention_count"] >= 2


def test_track_topic():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    topics = cm.track_topic("u1", "I'm worried about my job interview tomorrow.")
    assert "career" in topics or "anxiety" in topics
    profile = cm.get_user_profile("u1")
    assert len(profile["topics"]) >= 1


def test_topic_sentiment():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.track_topic("u1", "I'm really happy about my new relationship! It's wonderful.")
    profile = cm.get_user_profile("u1")
    love_topics = [t for t in profile["topics"] if t["topic"] == "love"]
    assert len(love_topics) >= 1
    assert love_topics[0]["sentiment"] == "positive"


def test_record_followup():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.record_followup("u1", "Check in about the job interview", due_days=3)
    followups = cm.get_pending_followups("u1")
    assert len(followups) == 1
    assert followups[0]["promise_text"] == "Check in about the job interview"
    assert followups[0]["status"] == "pending"


def test_fulfill_followup():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.record_followup("u1", "Test promise", due_days=1)
    followups = cm.get_pending_followups("u1")
    cm.fulfill_followup(followups[0]["id"])
    remaining = cm.get_pending_followups("u1")
    assert len(remaining) == 0


def test_log_card_and_patterns():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.log_card("u1", "The Tower", False, "celtic_cross", "position_1", 1)
    cm.log_card("u1", "The Tower", True, "past_present_future", "past", 2)
    cm.log_card("u1", "The Moon", False, "daily_pull", "single", 3)

    patterns = cm.get_card_patterns("u1")
    assert patterns["total_cards"] == 3
    assert len(patterns["recurring"]) >= 1
    assert patterns["most_drawn"] == "The Tower"
    assert patterns["reversal_ratio"] > 0


def test_log_timeline_event():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.log_timeline_event("u1", "job_change", "Started new job as a designer", "2026-03-01")
    profile = cm.get_user_profile("u1")
    assert len(profile["timeline"]) == 1
    assert profile["timeline"][0]["event_type"] == "job_change"


def test_empty_user_profile():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    profile = cm.get_user_profile("unknown_user")
    assert profile["user_id"] == "unknown_user"
    assert profile["entities"] == {}
    assert profile["topics"] == []
    assert profile["timeline"] == []
