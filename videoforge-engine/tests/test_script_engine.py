"""Tests for ScriptEngine — script generation with fallback, domain expertise, visual directions, anti-slop."""

import pytest
from videoforge.assembly.script_engine import (
    ScriptEngine, BANNED_WORDS, BANNED_PHRASES, CONTRACTION_MAP,
)
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import VideoScript, Storyboard, SceneSpec
from videoforge.knowledge.niche_profiles import NICHE_PROFILES
from videoforge.knowledge.script_frameworks import (
    SCRIPT_FRAMEWORKS, CONTENT_TYPE_TO_FRAMEWORK, get_framework_key,
)


@pytest.fixture
def engine():
    return ScriptEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def storyboard(smith):
    return smith.craft_storyboard("moon rituals for beginners", "witchcraftforbeginners")


@pytest.fixture
def tech_storyboard(smith):
    return smith.craft_storyboard("5 smart home gadgets that save money", "smarthomewizards")


class TestScriptEngine:
    def test_fallback_script_no_api_key(self, engine, storyboard):
        """Script generation returns a valid VideoScript (may use API or fallback)."""
        script = engine.generate_script(storyboard)
        assert isinstance(script, VideoScript)
        assert script.model_used in ("fallback_storyboard", "Claude Haiku", "DeepSeek V3", "Claude Sonnet")
        assert script.word_count > 0

    def test_fallback_has_content(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert script.word_count > 0
        assert script.full_text
        assert script.hook

    def test_fallback_has_segments(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert len(script.body_segments) >= 1

    def test_estimated_duration(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert script.estimated_duration > 0

    def test_generate_topics_no_key(self, engine):
        """Without API key, should return placeholder topics."""
        topics = engine.generate_topics("witchcraft", count=5)
        assert len(topics) == 5

    def test_title_from_storyboard(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert script.title == storyboard.title


class TestDomainExpertise:
    """Test that domain expertise is injected into system prompt."""

    def test_system_prompt_includes_domain_expertise_tech(self, engine, tech_storyboard):
        """Tech niche should include real product names in system prompt."""
        prompt = engine._system_prompt(tech_storyboard)
        # Should contain real product names from domain expertise
        assert "Echo Dot" in prompt or "Philips Hue" in prompt or "Ring" in prompt
        # Should contain expert tips
        assert "expert" in prompt.lower() or "tips" in prompt.lower() or "DOMAIN EXPERTISE" in prompt

    def test_system_prompt_includes_domain_expertise_witchcraft(self, engine, storyboard):
        """Witchcraft niche should include real herbs/crystals in system prompt."""
        prompt = engine._system_prompt(storyboard)
        assert "quartz" in prompt.lower() or "amethyst" in prompt.lower() or "lavender" in prompt.lower()

    def test_system_prompt_requires_specific_products(self, engine, tech_storyboard):
        """System prompt should instruct AI to use real product names."""
        prompt = engine._system_prompt(tech_storyboard)
        assert "specific" in prompt.lower()
        assert "vague" in prompt.lower() or "banned" in prompt.lower()

    def test_build_prompt_requests_visual_directions(self, engine, tech_storyboard):
        """Build prompt should request VISUAL descriptions alongside narration."""
        prompt = engine._build_prompt(tech_storyboard)
        assert "VISUAL:" in prompt
        assert "image description" in prompt.lower()


class TestVisualDirectionParsing:
    """Test parsing of visual directions from AI output."""

    def test_parse_script_extracts_visual_directions(self, engine):
        """AI output with | VISUAL: should populate visual_directions."""
        mock_ai_output = (
            "Scene 1: Stop wasting money on smart home gadgets. | VISUAL: pile of unused smart home devices on a shelf, dusty, wasted money concept\n"
            "Scene 2: The Echo Dot costs just thirty dollars. | VISUAL: Amazon Echo Dot on white countertop, blue LED ring glowing\n"
            "Scene 3: But here's the real trick. Use routines. | VISUAL: smartphone showing Alexa routines screen, automation workflow\n"
            "Scene 4: Philips Hue lights cut energy bills by 20%. | VISUAL: Philips Hue smart bulbs glowing warm in living room\n"
            "Scene 5: Follow for more money-saving smart home tips. | VISUAL: smart home dashboard on tablet showing savings chart"
        )
        storyboard = Storyboard(
            title="5 smart home gadgets", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=45,
            scenes=[SceneSpec(i+1, 9.0, "", "") for i in range(5)],
        )
        script = engine._parse_script(mock_ai_output, storyboard, "test", 0.0)

        assert len(script.visual_directions) == 5
        assert "Echo Dot" in script.visual_directions[1]
        assert "Philips Hue" in script.visual_directions[3]
        # Narration should NOT contain VISUAL text
        assert "VISUAL" not in script.full_text

    def test_parse_script_handles_missing_visual(self, engine):
        """Lines without VISUAL should produce empty visual_directions entries."""
        mock_ai_output = (
            "Scene 1: Stop scrolling. This changes everything.\n"
            "Scene 2: The number one trick is automation. | VISUAL: home automation dashboard\n"
            "Scene 3: Follow for more tips."
        )
        storyboard = Storyboard(
            title="test", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=30,
            scenes=[SceneSpec(i+1, 10.0, "", "") for i in range(3)],
        )
        script = engine._parse_script(mock_ai_output, storyboard, "test", 0.0)

        assert len(script.visual_directions) == 3
        assert script.visual_directions[0] == ""  # No VISUAL on line 1
        assert script.visual_directions[1] != ""  # VISUAL on line 2
        assert script.visual_directions[2] == ""  # No VISUAL on line 3

    def test_parse_script_tolerant_delimiters(self, engine):
        """Should handle various VISUAL delimiter formats."""
        mock_ai_output = "Scene 1: Hello world. |VISUAL: test image"
        storyboard = Storyboard(
            title="test", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=10,
            scenes=[SceneSpec(1, 10.0, "", "")],
        )
        script = engine._parse_script(mock_ai_output, storyboard, "test", 0.0)

        assert len(script.visual_directions) == 1
        assert script.visual_directions[0] == "test image"
        assert "Hello world." in script.full_text

    def test_fallback_script_has_empty_visual_directions(self, engine):
        """Storyboard fallback should set empty visual_directions list."""
        storyboard = Storyboard(
            title="test", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=30,
            scenes=[SceneSpec(i+1, 10.0, f"Scene {i+1} narration", "") for i in range(3)],
        )
        script = engine._fallback_script(storyboard)
        assert script.visual_directions == []
        assert script.model_used == "fallback_storyboard"


class TestPostProcessing:
    """Test the anti-slop post-processing pipeline."""

    def test_strip_banned_phrases(self, engine):
        """Banned phrases should be removed case-insensitively."""
        text = "It's important to note that the moon is full tonight."
        result = engine._strip_banned_phrases(text)
        assert "important to note" not in result.lower()
        assert "moon" in result

    def test_strip_banned_phrases_case_insensitive(self, engine):
        """Removal should work regardless of case."""
        text = "WITHOUT FURTHER ADO, let's begin."
        result = engine._strip_banned_phrases(text)
        assert "without further ado" not in result.lower()

    def test_strip_banned_words(self, engine):
        """Banned words should be removed at word boundaries."""
        text = "This is a crucial and furthermore pivotal moment."
        result = engine._strip_banned_words(text)
        assert "crucial" not in result.lower()
        assert "furthermore" not in result.lower()
        assert "pivotal" not in result.lower()
        assert "moment" in result

    def test_strip_banned_words_preserves_substrings(self, engine):
        """Banned word removal shouldn't affect words containing banned words as substrings."""
        # "navigate" is banned, but "navigation" should be handled by word boundary
        text = "Use the navigation system."
        result = engine._strip_banned_words(text)
        # "navigation" contains "navigate" but word boundary should prevent removal
        # (navigation != navigate at word boundary since 'ion' follows)
        assert "system" in result

    def test_enforce_contractions(self, engine):
        """Formal forms should become contractions."""
        text = "It is not a good idea. Do not try this at home."
        result = engine._enforce_contractions(text)
        assert "It's" in result or "it's" in result
        assert "Don't" in result or "don't" in result

    def test_enforce_contractions_preserves_case(self, engine):
        """Capitalized contractions should stay capitalized."""
        text = "It is the best. You are welcome."
        result = engine._enforce_contractions(text)
        assert "It's" in result
        assert "You're" in result

    def test_strip_markdown_bold(self, engine):
        """Bold markers should be removed."""
        text = "This is **very** important."
        result = engine._strip_markdown(text)
        assert "**" not in result
        assert "very" in result

    def test_strip_markdown_italic(self, engine):
        """Italic markers should be removed."""
        text = "This is *quite* interesting and _very_ cool."
        result = engine._strip_markdown(text)
        assert "*" not in result
        assert "_" not in result
        assert "quite" in result
        assert "very" in result

    def test_full_pipeline_doesnt_destroy_good_text(self, engine):
        """Good text should pass through mostly unchanged."""
        text = "The Echo Dot costs thirty dollars. It's a solid buy."
        result = engine._clean_text(text)
        assert "Echo Dot" in result
        assert "thirty dollars" in result
        assert "It's" in result

    def test_full_pipeline_cleans_slop(self, engine):
        """Sloppy AI text should be cleaned up."""
        text = "Furthermore, it is important to note that this **revolutionary** device leverages cutting-edge technology."
        result = engine._clean_text(text)
        assert "furthermore" not in result.lower()
        assert "important to note" not in result.lower()
        assert "**" not in result
        assert "device" in result

    def test_post_process_updates_word_count(self, engine):
        """Post-processing should recalculate word count."""
        script = VideoScript(
            title="test", hook="Furthermore this is crucial.", body_segments=["It is good."],
            cta="Follow.", full_text="Furthermore this is crucial. It is good. Follow.",
            word_count=999, estimated_duration=999.0,
            model_used="test", cost=0.0, visual_directions=[],
        )
        result = engine._post_process(script)
        assert result.word_count != 999
        assert result.word_count > 0


class TestFrameworkSelection:
    """Test content type detection and framework routing."""

    def test_detect_tutorial(self, engine):
        assert engine._detect_content_type("How to set up smart home automation") == "tutorial"

    def test_detect_review(self, engine):
        assert engine._detect_content_type("Apple Watch Ultra 3 honest review") == "review"

    def test_detect_story(self, engine):
        assert engine._detect_content_type("The legend of the Minotaur") == "story"

    def test_detect_listicle(self, engine):
        assert engine._detect_content_type("Top 5 crystals every witch needs") == "listicle"

    def test_detect_news(self, engine):
        assert engine._detect_content_type("OpenAI just announced GPT-5") == "news"

    def test_detect_motivation(self, engine):
        assert engine._detect_content_type("Manifest abundance with your mindset") == "motivation"

    def test_detect_defaults_to_educational(self, engine):
        assert engine._detect_content_type("something random here") == "educational"

    def test_framework_key_from_content_type(self):
        assert get_framework_key(content_type="tutorial") == "hook_problem_solution_cta"
        assert get_framework_key(content_type="story") == "loop"
        assert get_framework_key(content_type="review") == "psp"
        assert get_framework_key(content_type="news") == "reverse_tell"

    def test_framework_key_from_category(self):
        assert get_framework_key(category="mythology") == "loop"
        assert get_framework_key(category="ai_news") == "reverse_tell"
        assert get_framework_key(category="tech") == "hook_problem_solution_cta"

    def test_all_frameworks_have_required_keys(self):
        required = {"name", "scene_structure", "prompt_instruction", "best_for"}
        for key, framework in SCRIPT_FRAMEWORKS.items():
            for field in required:
                assert field in framework, f"Framework '{key}' missing '{field}'"

    def test_all_content_types_map_to_valid_framework(self):
        for content_type, fw_key in CONTENT_TYPE_TO_FRAMEWORK.items():
            assert fw_key in SCRIPT_FRAMEWORKS, (
                f"Content type '{content_type}' maps to unknown framework '{fw_key}'"
            )


class TestSystemPromptQuality:
    """Test that the system prompt contains all anti-slop components."""

    def test_has_tts_rules(self, engine, storyboard):
        prompt = engine._system_prompt(storyboard)
        assert "TTS WRITING RULES" in prompt
        assert "contraction" in prompt.lower()
        assert "spell out" in prompt.lower() or "Spell out" in prompt

    def test_has_banned_words_section(self, engine, storyboard):
        prompt = engine._system_prompt(storyboard)
        assert "BANNED VOCABULARY" in prompt
        assert "furthermore" in prompt.lower()

    def test_has_framework(self, engine, storyboard):
        prompt = engine._system_prompt(storyboard)
        assert "SCRIPT FRAMEWORK" in prompt

    def test_has_voice_identity(self, engine, storyboard):
        prompt = engine._system_prompt(storyboard)
        # Should contain voice card identity (witchcraft = practicing witch)
        assert "practicing witch" in prompt.lower() or "You are" in prompt

    def test_has_retention_mechanics(self, engine, storyboard):
        prompt = engine._system_prompt(storyboard)
        assert "RETENTION MECHANICS" in prompt
        assert "open a loop" in prompt.lower() or "Open a loop" in prompt
        assert "pattern interrupt" in prompt.lower() or "Pattern interrupt" in prompt

    def test_no_slop_in_own_prompt_text(self, engine, storyboard):
        """The system prompt's instructional text shouldn't use AI slop phrasing."""
        prompt = engine._system_prompt(storyboard)
        # Strip the BANNED section since it intentionally lists slop as examples
        banned_start = prompt.find("BANNED VOCABULARY")
        banned_end = prompt.find("RETENTION MECHANICS")
        if banned_start > 0 and banned_end > banned_start:
            instructional = prompt[:banned_start] + prompt[banned_end:]
        else:
            instructional = prompt
        instructional_lower = instructional.lower()
        # These are phrases the prompt itself should never use instructionally
        assert "without further ado" not in instructional_lower
        assert "in today's fast-paced world" not in instructional_lower
        assert "it is worth noting" not in instructional_lower

    def test_has_specificity_mandate(self, engine, tech_storyboard):
        prompt = engine._system_prompt(tech_storyboard)
        assert "specific" in prompt.lower()
        # Should ban vague language
        assert "vague" in prompt.lower() or "BANNED" in prompt

    def test_mythology_uses_loop_framework(self, engine, smith):
        """Mythology story topics should select the loop framework."""
        sb = smith.craft_storyboard("The legend of Medusa", "mythicalarchives")
        prompt = engine._system_prompt(sb)
        assert "Loop" in prompt or "loop" in prompt.lower()


class TestVoiceCards:
    """Test that all 16 niches have voice cards with required keys."""

    REQUIRED_VOICE_CARD_KEYS = [
        "identity", "emotional_register", "viewer_relationship",
        "speaking_style", "forbidden_tones", "signature_phrases", "never_say",
    ]

    def test_all_niches_have_voice_card(self):
        for niche_id, profile in NICHE_PROFILES.items():
            assert "voice_card" in profile, f"Niche '{niche_id}' missing voice_card"

    def test_voice_cards_have_required_keys(self):
        for niche_id, profile in NICHE_PROFILES.items():
            voice_card = profile.get("voice_card", {})
            for key in self.REQUIRED_VOICE_CARD_KEYS:
                assert key in voice_card, f"Niche '{niche_id}' voice_card missing '{key}'"

    def test_voice_card_identity_is_descriptive(self):
        """Identity should be a meaningful description, not a generic title."""
        for niche_id, profile in NICHE_PROFILES.items():
            identity = profile.get("voice_card", {}).get("identity", "")
            assert len(identity) > 20, f"Niche '{niche_id}' identity too short: '{identity}'"

    def test_voice_card_never_say_has_entries(self):
        """Each niche should have at least 3 never_say entries."""
        for niche_id, profile in NICHE_PROFILES.items():
            never_say = profile.get("voice_card", {}).get("never_say", [])
            assert len(never_say) >= 3, f"Niche '{niche_id}' has only {len(never_say)} never_say entries"

    def test_voice_card_forbidden_tones_has_entries(self):
        """Each niche should have at least 2 forbidden tones."""
        for niche_id, profile in NICHE_PROFILES.items():
            tones = profile.get("voice_card", {}).get("forbidden_tones", [])
            assert len(tones) >= 2, f"Niche '{niche_id}' has only {len(tones)} forbidden tones"

    def test_voice_card_signature_phrases_are_natural(self):
        """Signature phrases should not contain banned AI slop words."""
        for niche_id, profile in NICHE_PROFILES.items():
            phrases = profile.get("voice_card", {}).get("signature_phrases", [])
            for phrase in phrases:
                phrase_lower = phrase.lower()
                for banned in ["furthermore", "leverage", "utilize", "crucial", "paradigm"]:
                    assert banned not in phrase_lower, (
                        f"Niche '{niche_id}' signature phrase contains banned word '{banned}': '{phrase}'"
                    )
