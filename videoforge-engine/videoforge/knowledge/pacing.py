"""Pacing profiles by platform and niche — scene duration, cut frequency, energy curves."""

PACING_PROFILES = {
    # ── Platform-based pacing ──
    "youtube_shorts": {
        "name": "YouTube Shorts",
        "avg_scene_duration": 3.5,
        "max_scene_duration": 8,
        "min_scene_duration": 1.5,
        "cuts_per_minute": 14,
        "hook_window_seconds": 1.5,
        "energy_curve": "spike_sustain",  # High start, maintain
        "ideal_total_duration": (30, 60),
        "word_rate_wpm": 150,
    },
    "tiktok": {
        "name": "TikTok",
        "avg_scene_duration": 2.0,
        "max_scene_duration": 4,
        "min_scene_duration": 0.8,
        "cuts_per_minute": 22,
        "hook_window_seconds": 1.0,
        "energy_curve": "spike_sustain",
        "ideal_total_duration": (15, 60),
        "word_rate_wpm": 175,
    },
    "instagram_reels": {
        "name": "Instagram Reels",
        "avg_scene_duration": 2.5,
        "max_scene_duration": 5,
        "min_scene_duration": 1,
        "cuts_per_minute": 18,
        "hook_window_seconds": 1.5,
        "energy_curve": "spike_sustain",
        "ideal_total_duration": (15, 90),
        "word_rate_wpm": 160,
    },
    "youtube": {
        "name": "YouTube Standard",
        "avg_scene_duration": 5.0,
        "max_scene_duration": 15,
        "min_scene_duration": 2,
        "cuts_per_minute": 10,
        "hook_window_seconds": 5.0,
        "energy_curve": "wave",  # Rising and falling tension
        "ideal_total_duration": (120, 600),
        "word_rate_wpm": 150,
    },
    "facebook_reels": {
        "name": "Facebook Reels",
        "avg_scene_duration": 3.0,
        "max_scene_duration": 6,
        "min_scene_duration": 1.5,
        "cuts_per_minute": 15,
        "hook_window_seconds": 2.0,
        "energy_curve": "gradual_build",
        "ideal_total_duration": (30, 90),
        "word_rate_wpm": 155,
    },
    # ── Niche-based pacing overrides ──
    "witchcraft": {
        "name": "Witchcraft / Spiritual",
        "avg_scene_duration": 4.0,
        "max_scene_duration": 8,
        "min_scene_duration": 2,
        "cuts_per_minute": 10,
        "hook_window_seconds": 2.0,
        "energy_curve": "ritual_arc",  # Slow build, peak, gentle close
        "ideal_total_duration": (45, 90),
        "word_rate_wpm": 140,
    },
    "mythology": {
        "name": "Mythology / Documentary",
        "avg_scene_duration": 5.0,
        "max_scene_duration": 10,
        "min_scene_duration": 3,
        "cuts_per_minute": 8,
        "hook_window_seconds": 3.0,
        "energy_curve": "epic_arc",  # Dramatic build, climax, resolution
        "ideal_total_duration": (60, 180),
        "word_rate_wpm": 145,
    },
    "tech_review": {
        "name": "Tech / Product Review",
        "avg_scene_duration": 3.0,
        "max_scene_duration": 6,
        "min_scene_duration": 1.5,
        "cuts_per_minute": 15,
        "hook_window_seconds": 2.0,
        "energy_curve": "informational",  # Steady with punctuation
        "ideal_total_duration": (60, 120),
        "word_rate_wpm": 160,
    },
    "ai_news": {
        "name": "AI / Tech News",
        "avg_scene_duration": 2.5,
        "max_scene_duration": 5,
        "min_scene_duration": 1.5,
        "cuts_per_minute": 18,
        "hook_window_seconds": 1.5,
        "energy_curve": "spike_sustain",
        "ideal_total_duration": (30, 90),
        "word_rate_wpm": 170,
    },
    "lifestyle": {
        "name": "Lifestyle / Family",
        "avg_scene_duration": 3.5,
        "max_scene_duration": 7,
        "min_scene_duration": 2,
        "cuts_per_minute": 12,
        "hook_window_seconds": 2.0,
        "energy_curve": "conversational",  # Natural rhythm
        "ideal_total_duration": (45, 120),
        "word_rate_wpm": 150,
    },
}

# Map niche IDs to pacing profiles
NICHE_PACING_MAP = {
    "witchcraftforbeginners": "witchcraft",
    "moonrituallibrary": "witchcraft",
    "manifestandalign": "witchcraft",
    "mythicalarchives": "mythology",
    "smarthomewizards": "tech_review",
    "smarthomegearreviews": "tech_review",
    "pulsegearreviews": "tech_review",
    "wearablegearreviews": "tech_review",
    "aidiscoverydigest": "ai_news",
    "aiinactionhub": "ai_news",
    "clearainews": "ai_news",
    "wealthfromai": "ai_news",
    "bulletjournals": "lifestyle",
    "theconnectedhaven": "lifestyle",
    "familyflourish": "lifestyle",
    "celebrationseason": "lifestyle",
}


def get_pacing(platform: str = None, niche: str = None) -> dict:
    """Get pacing profile. Merges platform timing with niche voice/energy.

    Platform controls: ideal_total_duration, max/min_scene_duration, cuts_per_minute.
    Niche controls: word_rate_wpm, energy_curve, hook_window_seconds.
    """
    base = PACING_PROFILES.get(platform, PACING_PROFILES["youtube_shorts"])

    if niche and niche in NICHE_PACING_MAP:
        niche_key = NICHE_PACING_MAP[niche]
        niche_profile = PACING_PROFILES.get(niche_key)
        if niche_profile:
            merged = dict(base)
            merged["word_rate_wpm"] = niche_profile.get("word_rate_wpm", base["word_rate_wpm"])
            merged["energy_curve"] = niche_profile.get("energy_curve", base["energy_curve"])
            merged["hook_window_seconds"] = niche_profile.get("hook_window_seconds", base["hook_window_seconds"])
            return merged

    return base
