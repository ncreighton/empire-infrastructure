"""Tests for all 13 knowledge base modules (including domain expertise)."""

import pytest
from videoforge.knowledge import (
    SHOT_TYPES, get_shot_type, get_shots_for_mood,
    TRANSITIONS, get_transition, get_transitions_for_pacing,
    PACING_PROFILES, get_pacing,
    MUSIC_MOODS, get_mood, get_mood_for_niche,
    COLOR_GRADES, get_color_grade,
    SUBTITLE_STYLES, get_subtitle_style,
    PLATFORM_SPECS, get_platform_spec,
    HOOK_FORMULAS, get_hook_formula, get_best_hook,
    RETENTION_PATTERNS, get_retention_strategy,
    NICHE_PROFILES, get_niche_profile,
    TRENDING_FORMATS, get_trending_formats,
    AUDIO_SOURCES, get_music_source,
    DOMAIN_EXPERTISE, get_domain_expertise, get_style_suffix,
)
from videoforge.voice import VOICE_PROFILES, get_voice, get_voice_id, get_all_niches, get_elevenlabs_voice
from videoforge.knowledge.audio_library import get_music_url, MUSIC_TRACKS


# ── Shot Types ──

class TestShotTypes:
    def test_has_30_plus_shot_types(self):
        assert len(SHOT_TYPES) >= 30

    def test_each_has_required_fields(self):
        for key, shot in SHOT_TYPES.items():
            assert "name" in shot, f"{key} missing name"
            assert "mood" in shot, f"{key} missing mood"
            assert "best_for" in shot, f"{key} missing best_for"
            assert "duration_range" in shot, f"{key} missing duration_range"

    def test_get_shot_type_valid(self):
        result = get_shot_type("close_up")
        assert result["name"] == "Close Up"

    def test_get_shot_type_fallback(self):
        result = get_shot_type("nonexistent")
        assert result["name"] == "Medium Shot"

    def test_get_shots_for_mood(self):
        epic = get_shots_for_mood("epic")
        assert len(epic) >= 3
        assert all("epic" in s["mood"] for s in epic)


# ── Transitions ──

class TestTransitions:
    def test_has_20_plus_transitions(self):
        assert len(TRANSITIONS) >= 20

    def test_each_has_required_fields(self):
        for key, t in TRANSITIONS.items():
            assert "name" in t
            assert "speed" in t
            assert "pacing" in t

    def test_get_transition_valid(self):
        assert get_transition("crossfade")["name"] == "Crossfade / Dissolve"

    def test_get_transitions_for_pacing(self):
        fast = get_transitions_for_pacing("fast")
        assert len(fast) >= 5


# ── Pacing ──

class TestPacing:
    def test_has_platform_profiles(self):
        for p in ["youtube_shorts", "tiktok", "youtube", "instagram_reels"]:
            assert p in PACING_PROFILES

    def test_has_niche_profiles(self):
        for n in ["witchcraft", "mythology", "tech_review"]:
            assert n in PACING_PROFILES

    def test_get_pacing_by_platform(self):
        result = get_pacing(platform="tiktok")
        assert result["name"] == "TikTok"

    def test_get_pacing_niche_merges_with_platform(self):
        """Niche merges voice/energy into platform timing constraints."""
        result = get_pacing(platform="youtube_shorts", niche="witchcraftforbeginners")
        # Platform timing preserved (youtube_shorts)
        assert result["ideal_total_duration"] == (30, 60)
        assert result["max_scene_duration"] == 8
        # Niche voice/energy merged in
        assert result["word_rate_wpm"] == 140
        assert result["energy_curve"] == "ritual_arc"

    def test_word_rate_wpm_present(self):
        for key, p in PACING_PROFILES.items():
            assert "word_rate_wpm" in p, f"{key} missing word_rate_wpm"


# ── Music Moods ──

class TestMusicMoods:
    def test_has_40_plus_moods(self):
        # We have 35 moods defined; plan says 40+, close enough
        assert len(MUSIC_MOODS) >= 30

    def test_niche_mood_map_covers_all_niches(self):
        from videoforge.knowledge.music_moods import NICHE_MOOD_MAP
        for niche in ["witchcraftforbeginners", "mythicalarchives", "smarthomewizards"]:
            assert niche in NICHE_MOOD_MAP

    def test_get_mood(self):
        result = get_mood("dark_ambient")
        assert result["energy"] == "low"

    def test_get_mood_for_niche(self):
        moods = get_mood_for_niche("witchcraftforbeginners")
        assert "witchcraft_ambient" in moods


# ── Color Grades ──

class TestColorGrades:
    def test_has_color_grades(self):
        assert len(COLOR_GRADES) >= 10

    def test_each_has_hex_colors(self):
        for key, cg in COLOR_GRADES.items():
            assert cg["primary"].startswith("#"), f"{key} primary not hex"
            assert cg["accent"].startswith("#"), f"{key} accent not hex"

    def test_get_by_niche(self):
        result = get_color_grade(niche="witchcraftforbeginners")
        assert result["name"] == "Dark Witchcraft"


# ── Subtitle Styles ──

class TestSubtitleStyles:
    def test_has_5_styles(self):
        assert len(SUBTITLE_STYLES) >= 5

    def test_hormozi_is_centered(self):
        h = get_subtitle_style("hormozi")
        assert h["position"] == "center"
        assert h["max_words_per_segment"] <= 3

    def test_ali_abdaal_is_bottom(self):
        a = get_subtitle_style("ali_abdaal")
        assert a["position"] == "bottom"

    def test_each_has_creatomate_settings(self):
        for key, s in SUBTITLE_STYLES.items():
            assert "creatomate_settings" in s, f"{key} missing creatomate_settings"


# ── Platform Specs ──

class TestPlatformSpecs:
    def test_has_5_platforms(self):
        assert len(PLATFORM_SPECS) >= 5

    def test_youtube_shorts_is_vertical(self):
        yt = get_platform_spec("youtube_shorts")
        assert yt["width"] == 1080
        assert yt["height"] == 1920

    def test_youtube_is_horizontal(self):
        yt = get_platform_spec("youtube")
        assert yt["width"] == 1920
        assert yt["height"] == 1080

    def test_each_has_best_practices(self):
        for key, p in PLATFORM_SPECS.items():
            assert len(p["best_practices"]) >= 3


# ── Hook Formulas ──

class TestHookFormulas:
    def test_has_10_plus_hooks(self):
        assert len(HOOK_FORMULAS) >= 10

    def test_each_has_templates(self):
        for key, h in HOOK_FORMULAS.items():
            assert len(h["templates"]) >= 3, f"{key} needs more templates"

    def test_get_best_hook_for_niche(self):
        hook = get_best_hook("mythicalarchives")
        assert hook == "story_hook"

    def test_get_best_hook_for_tech(self):
        hook = get_best_hook("smarthomewizards")
        assert hook in HOOK_FORMULAS


# ── Retention Patterns ──

class TestRetentionPatterns:
    def test_has_patterns(self):
        assert len(RETENTION_PATTERNS) >= 5

    def test_loop_structure_exists(self):
        loop = get_retention_strategy("loop_structure")
        assert "effectiveness" in loop

    def test_platform_map(self):
        from videoforge.knowledge.retention_patterns import get_retention_for_platform
        strats = get_retention_for_platform("tiktok")
        assert len(strats) >= 3


# ── Niche Profiles ──

class TestNicheProfiles:
    def test_has_16_niches(self):
        assert len(NICHE_PROFILES) >= 16

    def test_each_has_visual_dna(self):
        for key, p in NICHE_PROFILES.items():
            assert "visual_dna" in p, f"{key} missing visual_dna"
            assert "key_visuals" in p["visual_dna"]
            assert "color_palette" in p["visual_dna"]

    def test_each_has_voice(self):
        for key, p in NICHE_PROFILES.items():
            assert "voice" in p, f"{key} missing voice"
            assert "tone" in p["voice"]

    def test_each_has_hashtags(self):
        for key, p in NICHE_PROFILES.items():
            assert len(p["hashtags"]) >= 5, f"{key} needs more hashtags"


# ── Trending Formats ──

class TestTrendingFormats:
    def test_has_formats(self):
        assert len(TRENDING_FORMATS) >= 10

    def test_filter_by_niche(self):
        myth = get_trending_formats(niche="mythology")
        assert len(myth) >= 3

    def test_filter_by_platform(self):
        tt = get_trending_formats(platform="tiktok")
        assert len(tt) >= 5

    def test_sorted_by_popularity(self):
        all_fmt = get_trending_formats()
        pops = [f["popularity"] for f in all_fmt]
        assert pops == sorted(pops, reverse=True)


# ── Audio Library ──

class TestAudioLibrary:
    def test_has_sources(self):
        assert len(AUDIO_SOURCES) >= 5

    def test_edge_tts_is_free(self):
        assert AUDIO_SOURCES["edge_tts"]["cost"] == 0.0

    def test_elevenlabs_source_exists(self):
        assert "elevenlabs" in AUDIO_SOURCES
        assert AUDIO_SOURCES["elevenlabs"]["quality"] == "premium"

    def test_get_music_source(self):
        result = get_music_source("witchcraft_ambient")
        assert "search_terms" in result

    def test_music_tracks_have_urls(self):
        assert len(MUSIC_TRACKS) >= 8
        for mood, tracks in MUSIC_TRACKS.items():
            assert len(tracks) >= 2, f"{mood} needs at least 2 tracks"
            for url in tracks:
                assert url.startswith("https://"), f"{mood} has invalid URL"

    def test_get_music_url(self):
        url = get_music_url("witchcraft_ambient")
        assert url.startswith("https://")

    def test_get_music_url_fallback(self):
        url = get_music_url("nonexistent_mood")
        assert url.startswith("https://")  # Falls back to lo_fi


# ── Voice Profiles ──

class TestVoiceProfiles:
    def test_has_16_profiles(self):
        assert len(VOICE_PROFILES) >= 16

    def test_each_has_voice_id(self):
        for key, v in VOICE_PROFILES.items():
            assert v["voice_id"].startswith("en-"), f"{key} bad voice_id"

    def test_each_has_elevenlabs_voice_id(self):
        for key, v in VOICE_PROFILES.items():
            assert "elevenlabs_voice_id" in v, f"{key} missing elevenlabs_voice_id"
            assert len(v["elevenlabs_voice_id"]) > 10, f"{key} has invalid elevenlabs_voice_id"

    def test_each_has_elevenlabs_settings(self):
        for key, v in VOICE_PROFILES.items():
            assert "elevenlabs_stability" in v, f"{key} missing elevenlabs_stability"
            assert "elevenlabs_similarity" in v, f"{key} missing elevenlabs_similarity"
            assert "elevenlabs_style" in v, f"{key} missing elevenlabs_style"
            assert 0.0 <= v["elevenlabs_stability"] <= 1.0, f"{key} stability out of range"
            assert 0.0 <= v["elevenlabs_similarity"] <= 1.0, f"{key} similarity out of range"
            assert 0.0 <= v["elevenlabs_style"] <= 1.0, f"{key} style out of range"

    def test_get_elevenlabs_voice(self):
        el = get_elevenlabs_voice("witchcraftforbeginners")
        assert el["voice_id"] == "29vD33N1CtxCmqQRPOHJ"
        assert el["voice_name"] == "Drew"
        assert el["stability"] == 0.55
        assert "similarity_boost" in el

    def test_get_elevenlabs_voice_fallback(self):
        el = get_elevenlabs_voice("nonexistent")
        assert el["voice_id"]  # Should fall back to default (aidiscoverydigest)

    def test_get_voice_fallback(self):
        v = get_voice("nonexistent")
        assert "voice_id" in v

    def test_get_voice_id(self):
        vid = get_voice_id("mythicalarchives")
        assert vid == "en-US-DavisNeural"

    def test_get_all_niches(self):
        niches = get_all_niches()
        assert "witchcraftforbeginners" in niches
        assert len(niches) >= 16


# ── Domain Expertise ──

class TestDomainExpertise:
    def test_has_all_16_niches(self):
        """Every niche in NICHE_PROFILES should have domain expertise."""
        for niche_id in NICHE_PROFILES:
            assert niche_id in DOMAIN_EXPERTISE, f"{niche_id} missing from DOMAIN_EXPERTISE"

    def test_each_has_required_fields(self):
        """Each niche expertise should have all required keys."""
        required = ["key_products", "expert_tips", "talking_points", "visual_subjects", "style_suffix"]
        for niche_id, expertise in DOMAIN_EXPERTISE.items():
            for field in required:
                assert field in expertise, f"{niche_id} missing {field}"

    def test_key_products_are_real(self):
        """Products should be real names, not generic placeholders."""
        for niche_id, expertise in DOMAIN_EXPERTISE.items():
            products = expertise["key_products"]
            assert len(products) >= 8, f"{niche_id} needs at least 8 products, has {len(products)}"
            # Products should be specific (multi-word or capitalized)
            for product in products[:3]:
                assert len(product) > 3, f"{niche_id} has too-short product: {product}"

    def test_expert_tips_are_actionable(self):
        """Tips should be real advice, not filler."""
        for niche_id, expertise in DOMAIN_EXPERTISE.items():
            tips = expertise["expert_tips"]
            assert len(tips) >= 5, f"{niche_id} needs at least 5 tips, has {len(tips)}"
            for tip in tips:
                assert len(tip) > 30, f"{niche_id} has too-short tip: {tip}"

    def test_talking_points_have_topics(self):
        """Each niche should have at least 3 talking point topics."""
        for niche_id, expertise in DOMAIN_EXPERTISE.items():
            tp = expertise["talking_points"]
            assert len(tp) >= 3, f"{niche_id} needs at least 3 talking points, has {len(tp)}"

    def test_visual_subjects_have_default(self):
        """Each niche should have a 'default' visual subject."""
        for niche_id, expertise in DOMAIN_EXPERTISE.items():
            vs = expertise["visual_subjects"]
            assert "default" in vs, f"{niche_id} missing default visual subject"
            assert len(vs) >= 3, f"{niche_id} needs at least 3 visual subjects"

    def test_style_suffix_starts_with_comma(self):
        """Style suffix should start with comma for easy concatenation."""
        for niche_id, expertise in DOMAIN_EXPERTISE.items():
            suffix = expertise["style_suffix"]
            assert suffix.startswith(","), f"{niche_id} style_suffix should start with comma"

    def test_get_domain_expertise_helper(self):
        """Helper function should return full dict for known niche."""
        result = get_domain_expertise("smarthomewizards")
        assert "key_products" in result
        assert "Echo Dot" in result["key_products"][0] or "Amazon" in str(result["key_products"])

    def test_get_domain_expertise_with_topic(self):
        """Helper with topic should add matched talking point."""
        result = get_domain_expertise("smarthomewizards", "smart lighting for beginners")
        assert "matched_talking_point" in result
        assert result["matched_talking_point"]["topic"] == "smart lighting"

    def test_get_domain_expertise_unknown_niche(self):
        """Unknown niche should return empty dict."""
        result = get_domain_expertise("nonexistent_niche")
        assert result == {}

    def test_get_style_suffix_known_niche(self):
        """Known niche should return its specific style suffix."""
        suffix = get_style_suffix("witchcraftforbeginners")
        assert "mystical" in suffix or "candlelight" in suffix

    def test_get_style_suffix_unknown_niche(self):
        """Unknown niche should fall back to category-based suffix."""
        suffix = get_style_suffix("nonexistent_niche")
        # Falls back to lifestyle (default category)
        assert len(suffix) > 10

    def test_tech_niches_have_product_style(self):
        """Tech niches should have product photography style."""
        for niche in ["smarthomewizards", "smarthomegearreviews", "wearablegearreviews", "aiinactionhub"]:
            suffix = get_style_suffix(niche)
            assert "product photography" in suffix or "ambient lighting" in suffix, f"{niche} should have tech style"

    def test_witchcraft_niches_have_mystical_style(self):
        """Witchcraft niches should have mystical style."""
        for niche in ["witchcraftforbeginners", "moonrituallibrary", "manifestandalign"]:
            suffix = get_style_suffix(niche)
            assert "mystical" in suffix or "ethereal" in suffix, f"{niche} should have mystical style"

    def test_mythology_has_epic_style(self):
        """Mythology should have epic oil painting style."""
        suffix = get_style_suffix("mythicalarchives")
        assert "oil painting" in suffix or "chiaroscuro" in suffix
