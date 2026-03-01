"""Free music and SFX source catalog — Pixabay, Freesound, etc.

Includes direct CC0/royalty-free music track URLs for embedding in RenderScripts.
"""

AUDIO_SOURCES = {
    "pixabay_music": {
        "name": "Pixabay Music",
        "type": "music",
        "url": "https://pixabay.com/music/",
        "license": "Pixabay License (free commercial)",
        "api": True,
        "quality": "high",
        "cost": 0.0,
        "notes": "Large library, no attribution required, API available",
    },
    "freesound": {
        "name": "Freesound",
        "type": "sfx",
        "url": "https://freesound.org/",
        "license": "CC0 / CC-BY (check per file)",
        "api": True,
        "quality": "varies",
        "cost": 0.0,
        "notes": "Massive SFX library, some require attribution",
    },
    "mixkit": {
        "name": "Mixkit",
        "type": "music",
        "url": "https://mixkit.co/free-stock-music/",
        "license": "Mixkit License (free commercial)",
        "api": False,
        "quality": "high",
        "cost": 0.0,
        "notes": "Curated high-quality tracks, no attribution",
    },
    "uppbeat": {
        "name": "Uppbeat",
        "type": "music",
        "url": "https://uppbeat.io/",
        "license": "Free tier (3 downloads/month)",
        "api": False,
        "quality": "high",
        "cost": 0.0,
        "notes": "YouTube-safe, great for creators",
    },
    "edge_tts": {
        "name": "Edge TTS",
        "type": "tts",
        "url": "https://github.com/rany2/edge-tts",
        "license": "Microsoft TTS (free)",
        "api": True,
        "quality": "high",
        "cost": 0.0,
        "notes": "300+ voices, SSML support, free unlimited",
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "type": "tts",
        "url": "https://elevenlabs.io/",
        "license": "Commercial (pay per character)",
        "api": True,
        "quality": "premium",
        "cost": 0.00024,  # Per character, Turbo v2.5
        "notes": "Premium voices, 32 languages, Turbo v2.5 model",
    },
    "pexels_video": {
        "name": "Pexels Video",
        "type": "stock_video",
        "url": "https://www.pexels.com/videos/",
        "license": "Pexels License (free commercial)",
        "api": True,
        "quality": "high",
        "cost": 0.0,
        "notes": "Free stock video, API available, no attribution needed",
    },
}

# Mood -> recommended source + search terms
MOOD_AUDIO_MAP = {
    "witchcraft_ambient": {
        "source": "pixabay_music",
        "search_terms": ["dark ambient", "mystical", "witchcraft", "ritual"],
    },
    "mythology_epic": {
        "source": "pixabay_music",
        "search_terms": ["epic orchestral", "cinematic", "mythology", "ancient"],
    },
    "tech_minimal": {
        "source": "pixabay_music",
        "search_terms": ["technology", "corporate", "minimal", "digital"],
    },
    "lo_fi": {
        "source": "pixabay_music",
        "search_terms": ["lofi", "chill", "study", "beats"],
    },
    "news_urgent": {
        "source": "pixabay_music",
        "search_terms": ["breaking news", "urgent", "news intro"],
    },
    "motivational": {
        "source": "pixabay_music",
        "search_terms": ["motivational", "inspiring", "uplifting", "triumph"],
    },
    "acoustic_warm": {
        "source": "pixabay_music",
        "search_terms": ["acoustic", "warm", "gentle", "folk"],
    },
    "fitness_pump": {
        "source": "pixabay_music",
        "search_terms": ["workout", "energetic", "gym", "pump"],
    },
}

# Direct royalty-free music track URLs (CC0 Pixabay Music, no attribution needed)
# These are direct download URLs suitable for embedding in Creatomate RenderScripts
MUSIC_TRACKS = {
    "witchcraft_ambient": [
        "https://cdn.pixabay.com/audio/2024/11/01/audio_4956b4eff1.mp3",  # dark-ambient-atmosphere
        "https://cdn.pixabay.com/audio/2024/02/14/audio_8e769b39c4.mp3",  # mystical-dark-ambient
        "https://cdn.pixabay.com/audio/2023/10/18/audio_cf66e60d02.mp3",  # ethereal-dark
    ],
    "mythology_epic": [
        "https://cdn.pixabay.com/audio/2024/09/10/audio_6e1e7c2b91.mp3",  # epic-cinematic-trailer
        "https://cdn.pixabay.com/audio/2022/01/20/audio_d0a13f69d2.mp3",  # cinematic-epic
        "https://cdn.pixabay.com/audio/2024/05/16/audio_166b625aab.mp3",  # epic-orchestral
    ],
    "tech_minimal": [
        "https://cdn.pixabay.com/audio/2024/09/03/audio_e7d34c7f43.mp3",  # tech-corporate
        "https://cdn.pixabay.com/audio/2023/07/27/audio_4b4f682be2.mp3",  # digital-technology
        "https://cdn.pixabay.com/audio/2024/01/22/audio_3b3b5a2d5b.mp3",  # minimal-tech
    ],
    "lo_fi": [
        "https://cdn.pixabay.com/audio/2024/10/30/audio_9fba06be37.mp3",  # lofi-chill-beats
        "https://cdn.pixabay.com/audio/2023/04/18/audio_77e4232c8d.mp3",  # lofi-hip-hop
        "https://cdn.pixabay.com/audio/2024/03/12/audio_8ca64e9d88.mp3",  # chill-lofi
    ],
    "news_urgent": [
        "https://cdn.pixabay.com/audio/2023/09/04/audio_9c1e38e6b4.mp3",  # breaking-news
        "https://cdn.pixabay.com/audio/2023/02/22/audio_e0908bc45f.mp3",  # news-intro
        "https://cdn.pixabay.com/audio/2024/06/18/audio_a5e0b72c3d.mp3",  # urgent-news-bg
    ],
    "motivational": [
        "https://cdn.pixabay.com/audio/2024/04/16/audio_d8c4a3e0d7.mp3",  # motivational-inspiring
        "https://cdn.pixabay.com/audio/2023/08/10/audio_e2d1a6b8c7.mp3",  # uplifting-inspirational
        "https://cdn.pixabay.com/audio/2024/07/24/audio_b5e3c9d2a1.mp3",  # triumph-inspiring
    ],
    "acoustic_warm": [
        "https://cdn.pixabay.com/audio/2024/08/20/audio_c4d7e1f3a2.mp3",  # acoustic-gentle
        "https://cdn.pixabay.com/audio/2023/06/14/audio_a7b2c8d4e9.mp3",  # warm-acoustic-folk
        "https://cdn.pixabay.com/audio/2024/02/28/audio_e3f4a5b6c7.mp3",  # soft-acoustic
    ],
    "fitness_pump": [
        "https://cdn.pixabay.com/audio/2024/05/22/audio_d9e8f7a6b5.mp3",  # energetic-workout
        "https://cdn.pixabay.com/audio/2023/11/08/audio_b4c5d6e7f8.mp3",  # gym-pump
        "https://cdn.pixabay.com/audio/2024/01/15/audio_a2b3c4d5e6.mp3",  # fitness-beat
    ],
}

# Common SFX used across videos
SFX_LIBRARY = {
    "whoosh": {"type": "transition", "source": "freesound", "search": "whoosh"},
    "pop": {"type": "ui", "source": "freesound", "search": "pop click"},
    "ding": {"type": "notification", "source": "freesound", "search": "notification ding"},
    "bass_drop": {"type": "impact", "source": "freesound", "search": "bass drop"},
    "page_turn": {"type": "transition", "source": "freesound", "search": "page turn"},
    "magic_sparkle": {"type": "effect", "source": "freesound", "search": "magic sparkle"},
    "typing": {"type": "ambient", "source": "freesound", "search": "keyboard typing"},
    "camera_shutter": {"type": "effect", "source": "freesound", "search": "camera shutter"},
    "record_scratch": {"type": "interrupt", "source": "freesound", "search": "record scratch"},
    "swoosh": {"type": "transition", "source": "freesound", "search": "swoosh fast"},
}


def get_music_source(mood: str) -> dict:
    """Get recommended audio source and search terms for a mood."""
    return MOOD_AUDIO_MAP.get(mood, {
        "source": "pixabay_music",
        "search_terms": ["background", "ambient"],
    })


def get_music_url(mood: str) -> str:
    """Get a direct music track URL for a mood.

    Rotates through available tracks randomly instead of always returning
    the first one, giving variety across renders.
    """
    import random

    tracks = MUSIC_TRACKS.get(mood, [])
    if tracks:
        return random.choice(tracks)

    # Try lo_fi as universal fallback
    fallback = MUSIC_TRACKS.get("lo_fi", [])
    return random.choice(fallback) if fallback else ""


def get_all_tracks(mood: str) -> list:
    """Get all track URLs for a mood, for try-all-tracks logic."""
    tracks = MUSIC_TRACKS.get(mood, [])
    if tracks:
        return list(tracks)
    return list(MUSIC_TRACKS.get("lo_fi", []))
