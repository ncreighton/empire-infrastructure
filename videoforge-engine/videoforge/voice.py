"""Voice profiles for 16 niches — Edge TTS voice IDs and speaking style."""

# Edge TTS voice map: niche -> {voice_id, name, rate, pitch, style_note}
# See: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support

VOICE_PROFILES = {
    "witchcraftforbeginners": {
        "voice_id": "en-US-AriaNeural",
        "name": "Aria",
        "rate": "-5%",
        "pitch": "-2st",
        "style": "calm",
        "style_note": "Mystical, warm, inviting — guides viewers into ritual space",
    },
    "smarthomewizards": {
        "voice_id": "en-US-GuyNeural",
        "name": "Guy",
        "rate": "+5%",
        "pitch": "0st",
        "style": "friendly",
        "style_note": "Enthusiastic tech reviewer, clear and upbeat",
    },
    "mythicalarchives": {
        "voice_id": "en-US-DavisNeural",
        "name": "Davis",
        "rate": "-8%",
        "pitch": "-3st",
        "style": "documentary",
        "style_note": "Deep, cinematic narrator — epic mythology storytelling",
    },
    "bulletjournals": {
        "voice_id": "en-US-JennyNeural",
        "name": "Jenny",
        "rate": "0%",
        "pitch": "+1st",
        "style": "gentle",
        "style_note": "Soft, creative, encouraging — artistic community vibe",
    },
    "wealthfromai": {
        "voice_id": "en-US-TonyNeural",
        "name": "Tony",
        "rate": "+8%",
        "pitch": "0st",
        "style": "confident",
        "style_note": "Fast-paced hustle energy, authoritative on AI money topics",
    },
    "aidiscoverydigest": {
        "voice_id": "en-US-JasonNeural",
        "name": "Jason",
        "rate": "+3%",
        "pitch": "0st",
        "style": "newscast",
        "style_note": "News anchor delivery — breaking AI developments",
    },
    "aiinactionhub": {
        "voice_id": "en-US-BrandonNeural",
        "name": "Brandon",
        "rate": "+5%",
        "pitch": "+1st",
        "style": "tutorial",
        "style_note": "Practical, step-by-step tutorial energy",
    },
    "pulsegearreviews": {
        "voice_id": "en-US-AndrewNeural",
        "name": "Andrew",
        "rate": "+3%",
        "pitch": "-1st",
        "style": "review",
        "style_note": "Honest reviewer, fitness-forward, data-driven",
    },
    "wearablegearreviews": {
        "voice_id": "en-US-EricNeural",
        "name": "Eric",
        "rate": "+2%",
        "pitch": "0st",
        "style": "review",
        "style_note": "Tech reviewer, balanced and thorough",
    },
    "smarthomegearreviews": {
        "voice_id": "en-US-RogerNeural",
        "name": "Roger",
        "rate": "+3%",
        "pitch": "-1st",
        "style": "expert",
        "style_note": "Hands-on expert, trustworthy product breakdown",
    },
    "clearainews": {
        "voice_id": "en-US-SteffanNeural",
        "name": "Steffan",
        "rate": "+5%",
        "pitch": "0st",
        "style": "newscast",
        "style_note": "Fast news delivery, clear AI explanations",
    },
    "theconnectedhaven": {
        "voice_id": "en-US-SaraNeural",
        "name": "Sara",
        "rate": "-3%",
        "pitch": "+2st",
        "style": "warm",
        "style_note": "Warm family voice, relatable and supportive",
    },
    "manifestandalign": {
        "voice_id": "en-US-AmberNeural",
        "name": "Amber",
        "rate": "-5%",
        "pitch": "+1st",
        "style": "meditation",
        "style_note": "Soothing manifestation guide, positive affirmations",
    },
    "familyflourish": {
        "voice_id": "en-US-MichelleNeural",
        "name": "Michelle",
        "rate": "0%",
        "pitch": "+1st",
        "style": "conversational",
        "style_note": "Friendly mom energy, practical parenting tips",
    },
    "moonrituallibrary": {
        "voice_id": "en-US-AriaNeural",
        "name": "Aria",
        "rate": "-10%",
        "pitch": "-2st",
        "style": "meditative",
        "style_note": "Deep ritual guide, slow and intentional",
    },
    "celebrationseason": {
        "voice_id": "en-US-JennyNeural",
        "name": "Jenny",
        "rate": "+5%",
        "pitch": "+2st",
        "style": "excited",
        "style_note": "Festive, high-energy party planning voice",
    },
}


def get_voice(niche: str) -> dict:
    """Get voice profile for a niche. Falls back to default."""
    return VOICE_PROFILES.get(niche, VOICE_PROFILES["aidiscoverydigest"])


def get_voice_id(niche: str) -> str:
    """Get just the Edge TTS voice ID string."""
    return get_voice(niche)["voice_id"]


def get_all_niches() -> list:
    """Return all niche keys with voice profiles."""
    return list(VOICE_PROFILES.keys())
