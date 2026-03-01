"""
elevenlabs-tts -- ElevenLabs text-to-speech wrapper with voice profiles.
Extracted from videoforge-engine/videoforge/assembly/audio_engine.py.

Provides:
- generate_speech(): single text-to-speech generation
- generate_scenes(): batch TTS for video storyboard scenes
- upload_to_host(): upload audio to catbox.moe or tmpfiles.org
- measure_duration(): actual MP3 duration measurement
- estimate_duration(): voice-aware duration estimation
- VOICE_PROFILES: niche-specific voice mappings with tuned settings

Key lessons from VideoForge production:
- Turbo v2.5 is the best cost/quality balance
- Voice-specific WPM varies significantly (135-160 range)
- Catbox.moe gives permanent direct-download URLs
- Base64 data URIs do NOT work with Creatomate
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List

log = logging.getLogger(__name__)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "eleven_turbo_v2_5"

# Niche voice profiles with tuned voice settings.
# voice_id, name, stability, similarity_boost, style calibrated per niche.
VOICE_PROFILES: Dict[str, Dict] = {
    "witchcraft": {
        "voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew",
        "style": "mystical-warm",
        "stability": 0.50, "similarity_boost": 0.75, "style_weight": 0.4,
    },
    "mythology": {
        "voice_id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Dave",
        "style": "scholarly-wonder",
        "stability": 0.55, "similarity_boost": 0.70, "style_weight": 0.3,
    },
    "smart-home": {
        "voice_id": "nPczCjzI2devNBz1zQrb", "name": "Brian",
        "style": "tech-authority",
        "stability": 0.60, "similarity_boost": 0.80, "style_weight": 0.2,
    },
    "ai-news": {
        "voice_id": "Yko7PKs96k2ssGkDz4Mc", "name": "Henry",
        "style": "forward-analyst",
        "stability": 0.55, "similarity_boost": 0.75, "style_weight": 0.3,
    },
    "default": {
        "voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew",
        "style": "neutral",
        "stability": 0.50, "similarity_boost": 0.75, "style_weight": 0.0,
    },
}

# Voice-specific words-per-minute -- calibrated from actual ElevenLabs output.
# Slower voices accumulate timing errors if we assume a flat 150 WPM.
VOICE_WPM: Dict[str, int] = {
    "Drew": 140, "Dave": 135, "Brian": 155, "Henry": 150,
    "Daniel": 148, "Giovanni": 138, "Alice": 142, "Adam": 160,
    "Patrick": 152, "Harry": 150, "Rachel": 145, "Glinda": 138,
    "Grace": 142, "default": 150,
}


def _get_api_key(api_key: Optional[str] = None) -> str:
    """Resolve ElevenLabs API key from argument or environment."""
    key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise ValueError(
            "ELEVENLABS_API_KEY not set. Pass api_key or set env var."
        )
    return key


def get_voice_profile(niche: str) -> Dict:
    """Get the recommended voice profile for a niche."""
    return VOICE_PROFILES.get(niche, VOICE_PROFILES["default"])


def generate_speech(
    text: str,
    niche: str = "default",
    output_path: Optional[str] = None,
    model_id: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate speech from text using ElevenLabs API.

    Args:
        text: Text to convert to speech
        niche: Niche key for voice profile lookup
        output_path: Optional path to save the MP3 file
        model_id: ElevenLabs model (default: eleven_turbo_v2_5)
        api_key: API key (falls back to ELEVENLABS_API_KEY env var)

    Returns:
        MP3 audio bytes.
    """
    import requests

    key = _get_api_key(api_key)
    profile = get_voice_profile(niche)
    voice_id = profile["voice_id"]

    resp = requests.post(
        f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}"
        f"?output_format=mp3_44100_128",
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": profile.get("stability", 0.5),
                "similarity_boost": profile.get("similarity_boost", 0.75),
                "style": profile.get("style_weight", 0.0),
                "use_speaker_boost": True,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    audio_bytes = resp.content

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(audio_bytes)
        log.info("Audio saved: %s (%d bytes)", output_path, len(audio_bytes))

    return audio_bytes


def upload_to_host(file_path: str) -> str:
    """Upload an audio file to a public hosting service.

    Tries catbox.moe first (permanent), then tmpfiles.org (temporary).
    Returns a publicly accessible direct-download URL, or empty string.
    """
    import requests

    if not file_path or not os.path.exists(file_path):
        return ""

    # Try catbox.moe (permanent hosting, direct links)
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (
                    os.path.basename(file_path), f, "audio/mpeg"
                )},
                timeout=60,
            )
            resp.raise_for_status()
            url = resp.text.strip()
            if url.startswith("http"):
                log.info("Catbox upload: %s -> %s", file_path, url)
                return url
    except Exception as e:
        log.warning("Catbox upload failed: %s", e)

    # Try tmpfiles.org (temporary, auto-expires)
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": (
                    os.path.basename(file_path), f, "audio/mpeg"
                )},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "success":
                view_url = data["data"]["url"]
                dl_url = view_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                log.info("tmpfiles upload: %s -> %s", file_path, dl_url)
                return dl_url
    except Exception as e:
        log.warning("tmpfiles upload failed: %s", e)

    return ""


def measure_duration(file_path: str) -> float:
    """Measure actual MP3 duration in seconds.

    Tries mutagen first (most accurate), falls back to file-size estimation.
    """
    if not file_path or not os.path.exists(file_path):
        return 0.0

    try:
        from mutagen.mp3 import MP3
        audio = MP3(file_path)
        return audio.info.length
    except Exception:
        pass

    try:
        file_size = os.path.getsize(file_path)
        bitrate = 128_000  # 128 kbps in bits/s
        return (file_size * 8) / bitrate
    except Exception:
        return 0.0


def estimate_duration(text: str, voice_name: str = "",
                      wpm: Optional[int] = None) -> float:
    """Estimate TTS duration in seconds from text length.

    Uses voice-specific WPM when voice_name is provided.
    """
    if wpm is None:
        wpm = VOICE_WPM.get(voice_name, VOICE_WPM["default"])
    word_count = len(text.split())
    return (word_count / wpm) * 60


def estimate_cost(text: str) -> float:
    """Estimate ElevenLabs cost for text.

    Turbo v2.5 pricing: approx $0.00024 per character.
    """
    return len(text) * 0.00024
