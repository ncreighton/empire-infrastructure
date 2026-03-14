"""Tests for the Context Engine."""

import tempfile
from pathlib import Path
from moneyclaw.memory import Memory
from moneyclaw.services.luna.context import ContextEngine, ContextBundle
from moneyclaw.services.luna.conv_memory import ConversationalMemory


def _tmp_memory():
    return Memory(db_path=Path(tempfile.mktemp(suffix=".db")))


def test_context_engine_init():
    mem = _tmp_memory()
    ce = ContextEngine(mem)
    assert ce.grimoire is not None
    assert ce.conv_memory is not None


def test_build_context_empty_user():
    mem = _tmp_memory()
    ce = ContextEngine(mem)
    bundle = ce.build_context("new_user", "Hello Luna!", "web")
    assert isinstance(bundle, ContextBundle)
    assert bundle.moon_data["phase"]  # Should have moon phase
    assert bundle.seasonal_data["season"]
    assert bundle.user_profile["user_id"] == "new_user"
    assert bundle.card_patterns["total_cards"] == 0


def test_build_context_with_user_data():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.extract_and_store("u1", "My husband John is stressed about work.")
    cm.log_card("u1", "The Tower", False, "daily_pull", "single")
    cm.log_card("u1", "The Tower", True, "celtic_cross", "position_1")
    cm.record_followup("u1", "Check on the job situation", due_days=1)

    ce = ContextEngine(mem)
    bundle = ce.build_context("u1", "How is my career looking?", "telegram")
    assert len(bundle.user_profile["entities"].get("person", [])) >= 1
    assert bundle.card_patterns["total_cards"] == 2
    assert len(bundle.pending_followups) >= 1
    assert "career" in bundle.detected_topics


def test_build_context_topic_detection():
    mem = _tmp_memory()
    ce = ContextEngine(mem)
    bundle = ce.build_context("u1", "I'm worried about my relationship with my partner", "web")
    assert "love" in bundle.detected_topics or "anxiety" in bundle.detected_topics


def test_build_context_grimoire_integration():
    mem = _tmp_memory()
    ce = ContextEngine(mem)
    bundle = ce.build_context("u1", "I need healing energy today", "web")
    assert "health" in bundle.detected_topics
    # Should have grimoire correspondences for healing
    if bundle.grimoire_data:
        assert len(bundle.grimoire_data.get("herbs", [])) > 0 or bundle.grimoire_data.get("intention")


def test_format_for_prompt_empty():
    mem = _tmp_memory()
    ce = ContextEngine(mem)
    bundle = ContextBundle()
    result = ce.format_for_prompt(bundle)
    assert result == ""  # Empty bundle produces empty string


def test_format_for_prompt_with_data():
    mem = _tmp_memory()
    cm = ConversationalMemory(mem)
    cm.record_followup("u1", "Check in about the interview", due_days=0)
    cm.log_card("u1", "The Tower", False, "daily_pull", "single")
    cm.log_card("u1", "The Tower", True, "celtic_cross", "pos_1")

    ce = ContextEngine(mem)
    bundle = ce.build_context("u1", "Hello!", "web")
    formatted = ce.format_for_prompt(bundle)
    assert "Follow-ups" in formatted or "Card Patterns" in formatted


def test_context_bundle_defaults():
    bundle = ContextBundle()
    assert bundle.moon_data == {}
    assert bundle.seasonal_data == {}
    assert bundle.user_profile == {}
    assert bundle.card_patterns == {}
    assert bundle.pending_followups == []
    assert bundle.recent_history == []
    assert bundle.proactive_triggers == []
    assert bundle.detected_topics == []


def test_recent_history():
    mem = _tmp_memory()
    mem.log_interaction("u1", "web", "reading", service="tarot", spread_type="daily_pull")
    mem.log_interaction("u1", "telegram", "reading", service="tarot", spread_type="celtic_cross")

    ce = ContextEngine(mem)
    bundle = ce.build_context("u1", "Hi!", "web")
    assert len(bundle.recent_history) >= 1
