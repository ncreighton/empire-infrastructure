"""Tests for the Luna Engine — master orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from moneyclaw.memory import Memory
from moneyclaw.services.luna.luna_engine import LunaEngine, LunaResponse


def _tmp_memory():
    return Memory(db_path=Path(tempfile.mktemp(suffix=".db")))


def _mock_claude_response(text="I see you, darling. The cards whisper tonight under the full moon. Try carrying an amethyst crystal and light a lavender candle this evening. Journal about what's stirring inside you. Come back and let me know how the ritual goes. Blessed be. 🌙"):
    """Create a mock Claude API response."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=text)]
    mock_msg.usage = MagicMock(input_tokens=500, output_tokens=100)
    return mock_msg


def test_luna_engine_init():
    mem = _tmp_memory()
    engine = LunaEngine(mem)
    assert engine.memory is mem
    assert engine.conv_memory is not None
    assert engine.context_engine is not None
    assert engine.persona is not None
    assert engine.quality_gate is not None


def test_respond_chat(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response()
    engine._client = mock_client

    result = engine.respond("u1", "Hi Luna, I'm feeling anxious today", "web")
    assert isinstance(result, LunaResponse)
    assert result.text  # Non-empty response
    assert result.handler == "chat"
    assert result.mode  # Has a mode set


def test_respond_routes_reading(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(
        "I sense you're ready for a reading, darling. The cards are eager tonight under the full moon. Try a daily card pull to start, or if you need deeper guidance, the Celtic Cross spread reveals layers. Come back and let me know which calls to your spirit. 🌙"
    )
    engine._client = mock_client

    result = engine.respond("u1", "Can you draw a card for me?", "web")
    assert result.handler == "reading"


def test_respond_routes_spell(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(
        "A protection spell, darling — the universe heard you. Gather rosemary and sage under tonight's full moon. Light a black candle and carry black tourmaline close. Try cleansing your space tonight before bed. Come back and let me know how the ritual feels. 🌙"
    )
    engine._client = mock_client

    result = engine.respond("u1", "I need a protection spell", "web")
    assert result.handler == "spell"


def test_respond_routes_knowledge(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response(
        "Amethyst, darling — the stone of intuition! Carry it during meditation tonight for deeper insight. Try placing it under your pillow this week. Come back and tell me about your dreams. 🌙"
    )
    engine._client = mock_client

    result = engine.respond("u1", "Tell me about amethyst crystal properties", "web")
    assert result.handler == "knowledge"


def test_entity_extraction_during_response(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response()
    engine._client = mock_client

    engine.respond("u1", "My husband John is stressed about his new job", "web")

    # Check that entities were extracted
    from moneyclaw.services.luna.conv_memory import ConversationalMemory
    cm = ConversationalMemory(mem)
    profile = cm.get_user_profile("u1")
    people = profile["entities"].get("person", [])
    assert any(p["entity_name"] == "John" for p in people)


def test_quality_gate_triggers_retry(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    # First response is generic (low quality), second is good
    call_count = [0]
    def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_claude_response("I understand your concern. Here are some suggestions. I hope this helps.")
        return _mock_claude_response()

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = mock_create
    engine._client = mock_client

    result = engine.respond("u1", "I'm scared", "web")
    assert result.retried is True
    assert call_count[0] >= 2


def test_respond_logs_interaction(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response()
    engine._client = mock_client

    engine.respond("u1", "Hello Luna", "telegram")

    with mem._conn() as conn:
        rows = conn.execute(
            "SELECT * FROM interactions WHERE customer_id = 'u1' AND service = 'luna_engine'"
        ).fetchall()
    assert len(rows) >= 1


def test_fallback_response():
    mem = _tmp_memory()
    engine = LunaEngine(mem)
    text = engine._fallback_response()
    # Fallback always mentions the current moon phase name and "darling"
    assert "darling" in text.lower()
    assert len(text) > 30


def test_extract_intention():
    mem = _tmp_memory()
    engine = LunaEngine(mem)
    assert engine._extract_intention("I need a spell for protection") == "protection"
    assert engine._extract_intention("Help me with love") == "love"
    assert engine._extract_intention("Can you help with money stuff") == "money"
    assert engine._extract_intention("I feel unwell and need healing") == "healing"


def test_model_routing(monkeypatch):
    mem = _tmp_memory()
    engine = LunaEngine(mem)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_claude_response()
    engine._client = mock_client

    # Chat should use haiku
    result = engine.respond("u1", "Hi Luna", "web")
    call_args = mock_client.messages.create.call_args
    assert "haiku" in call_args.kwargs.get("model", "") or "haiku" in str(call_args)
