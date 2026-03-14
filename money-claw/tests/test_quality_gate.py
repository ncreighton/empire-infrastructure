"""Tests for the Response Quality Gate."""

from moneyclaw.services.luna.quality_gate import ResponseQualityGate


def test_quality_gate_init():
    qg = ResponseQualityGate()
    assert qg.threshold == 65


def test_high_quality_response():
    qg = ResponseQualityGate()
    response = (
        "I see you, darling. The cards are whispering something important tonight. "
        "Under this Full Moon, your energy is amplified — carry an amethyst crystal close "
        "to your heart this week. Try lighting a lavender candle before bed tonight and "
        "journal about what's weighing on you. The universe is working in your favor, "
        "even when it doesn't feel like it. Come back and let me know how the ritual goes. 🌙"
    )
    score, deductions = qg.score(response, "I'm feeling lost")
    assert score >= 65
    assert qg.passes(response, "I'm feeling lost")


def test_generic_response_fails():
    qg = ResponseQualityGate()
    response = (
        "I understand your concern. That's a great question. "
        "Here are some suggestions for you. I hope this helps. "
        "Feel free to ask if you have more questions."
    )
    score, deductions = qg.score(response, "Help me")
    assert score < 65
    assert not qg.passes(response, "Help me")


def test_forbidden_phrase_detection():
    qg = ResponseQualityGate()
    response = "As an AI, I cannot predict the future, but this is for entertainment purposes."
    score, deductions = qg.score(response, "Tell me my future")
    forbidden_deductions = [d for d in deductions if d["reason"] == "forbidden phrase"]
    assert len(forbidden_deductions) >= 1
    assert score < 70


def test_luna_voice_markers_boost():
    qg = ResponseQualityGate()
    # Same content with and without Luna markers
    generic = "You should try meditating tonight. It might help with your stress."
    luna = "Darling, the cards suggest you try meditating tonight under the full moon. The universe is supporting your healing. Come back and tell me how it goes."
    score_generic, _ = qg.score(generic, "I'm stressed")
    score_luna, _ = qg.score(luna, "I'm stressed")
    assert score_luna > score_generic


def test_empathy_for_distressed_user():
    qg = ResponseQualityGate()
    # No empathy response to distressed user
    response = "You should try some meditation techniques."
    score, deductions = qg.score(response, "I'm so scared and alone right now")
    empathy_deductions = [d for d in deductions if d["dimension"] == "empathy"]
    assert len(empathy_deductions) >= 1


def test_depth_length_check():
    qg = ResponseQualityGate()
    short_response = "Yes, that sounds good."
    score, deductions = qg.score(short_response, "Tell me more",
                                  context={"depth_level": 3})
    length_deductions = [d for d in deductions if "too short" in d.get("reason", "")]
    assert len(length_deductions) >= 1


def test_actionability_scoring():
    qg = ResponseQualityGate()
    # No actionable advice
    vague = "The stars are aligned in an interesting way. Your energy is shifting."
    score_vague, _ = qg.score(vague, "What should I do?")

    # With actionable advice
    actionable = (
        "Try lighting a white candle tonight and meditating for 10 minutes. "
        "Carry a clear quartz crystal this week. Journal before bed about "
        "your intentions. Come back next week and let me know how it went. 🌙"
    )
    score_action, _ = qg.score(actionable, "What should I do?")
    assert score_action > score_vague


def test_suggest_improvements():
    qg = ResponseQualityGate()
    generic = "I understand your concern. Here are some suggestions."
    suggestions = qg.suggest_improvements(generic, "Help me")
    assert len(suggestions) >= 1


def test_knowledge_scoring():
    qg = ResponseQualityGate()
    # With spiritual references
    rich = (
        "The Empress card is calling to you, darling. Work with rose quartz and "
        "lavender during this full moon. Try a grounding ritual tonight. "
        "Come back and share your experience. 🌙"
    )
    score_rich, _ = qg.score(rich, "I need guidance")

    # Without spiritual references
    poor = "Things will work out. Stay positive. Everything happens for a reason."
    score_poor, _ = qg.score(poor, "I need guidance")
    assert score_rich > score_poor


def test_personalization_scoring():
    qg = ResponseQualityGate()
    response = "The cards show a path forward for you."
    context = {
        "user_profile": {
            "entities": {
                "person": [{"entity_name": "Sarah"}],
            }
        }
    }
    score, deductions = qg.score(response, "Help me", context)
    personalization_deductions = [d for d in deductions if "personalization" in d.get("reason", "")]
    assert len(personalization_deductions) >= 1


def test_perfect_score_possible():
    qg = ResponseQualityGate()
    response = (
        "I see you, darling. The Tower card has been appearing in your readings lately, "
        "and under this full moon, its message is clear: transformation is calling. "
        "I understand how overwhelming that can feel, but you are not alone in this. "
        "Try carrying an amethyst crystal tonight and light a lavender candle before bed. "
        "Journal about what needs to crumble so something beautiful can grow. "
        "The waning moon this week supports release work. "
        "Come back tomorrow and let me know what surfaces. Blessed be. 🌙"
    )
    score, deductions = qg.score(response, "I'm going through big changes")
    assert score >= 70
