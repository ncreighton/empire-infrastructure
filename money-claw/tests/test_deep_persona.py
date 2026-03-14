"""Tests for the Deep Persona Engine."""

from moneyclaw.services.luna.deep_persona import DeepPersona, LunaState, MODES


def test_initial_state():
    dp = DeepPersona()
    assert dp.state.mode == "supportive"
    assert dp.state.emotional_tone == "warm"
    assert dp.state.depth_level == 2


def test_detect_nurturing_mode():
    dp = DeepPersona()
    state = dp.detect_mode("I'm crying and feeling so alone right now")
    assert state.mode == "nurturing"
    assert state.emotional_tone in MODES["nurturing"]["tone_options"]


def test_detect_fierce_mode():
    dp = DeepPersona()
    state = dp.detect_mode("I've had enough of this toxic relationship")
    assert state.mode == "fierce"


def test_detect_ceremonial_mode():
    dp = DeepPersona()
    state = dp.detect_mode("I want to cast a circle and do a ritual tonight")
    assert state.mode == "ceremonial"


def test_depth_increases():
    dp = DeepPersona()
    dp.detect_mode("Tell me more about what that really means for my life")
    assert dp.state.depth_level >= 3


def test_depth_decreases():
    dp = DeepPersona()
    dp._state.depth_level = 3
    dp.detect_mode("Just curious about something quick")
    assert dp.state.depth_level <= 2


def test_get_voice_modifiers():
    dp = DeepPersona()
    dp._state.mode = "wise"
    mods = dp.get_voice_modifiers()
    assert mods["mode"] == "wise"
    assert mods["mode_description"] == MODES["wise"]["description"]
    assert mods["depth"] == dp.state.depth_level


def test_build_system_prompt_basic():
    dp = DeepPersona()
    prompt = dp.build_system_prompt()
    assert "Luna Moonshadow" in prompt
    assert "supportive" in prompt.lower() or "Supportive" in prompt
    assert "NEVER" in prompt  # Forbidden phrases section
    assert "Moon Phase" in prompt


def test_build_system_prompt_with_user_profile():
    dp = DeepPersona()
    profile = {
        "entities": {
            "person": [{"entity_name": "John", "context": "husband"}],
            "pet": [{"entity_name": "Luna", "context": "cat"}],
        },
        "topics": [
            {"topic": "love", "sentiment": "positive"},
            {"topic": "career", "sentiment": "negative"},
        ],
        "timeline": [
            {"description": "Started new job", "event_type": "job_change"},
        ],
    }
    prompt = dp.build_system_prompt(user_profile=profile)
    assert "John" in prompt
    assert "Luna" in prompt
    assert "love" in prompt.lower()
    assert "Started new job" in prompt


def test_build_system_prompt_with_context():
    dp = DeepPersona()
    context = "## Pending Follow-ups\n- Check in about the interview"
    prompt = dp.build_system_prompt(context_block=context)
    assert "Pending Follow-ups" in prompt
    assert "interview" in prompt


def test_transition():
    dp = DeepPersona()
    assert dp.state.mode == "supportive"
    dp.transition("I need to cast a circle tonight")
    assert dp.state.mode == "ceremonial"


def test_reset():
    dp = DeepPersona()
    dp._state.mode = "fierce"
    dp._state.depth_level = 4
    dp.reset()
    assert dp.state.mode == "supportive"
    assert dp.state.depth_level == 2
