"""Free music and SFX source catalog — Pixabay, Freesound, etc."""

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
