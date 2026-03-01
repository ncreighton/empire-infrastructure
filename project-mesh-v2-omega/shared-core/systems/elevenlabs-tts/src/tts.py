"""
elevenlabs-tts — ElevenLabs text-to-speech wrapper with voice profiles.
Used by VideoForge and 3D-Print-Forge for narration generation.
"""

import os
import logging
from typing import Optional, Dict
from pathlib import Path

log = logging.getLogger(__name__)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"

# Niche voice profiles (voice_id -> description)
VOICE_PROFILES = {
    "witchcraft": {"voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew", "style": "mystical-warm"},
    "mythology": {"voice_id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Dave", "style": "scholarly-wonder"},
    "smart-home": {"voice_id": "nPczCjzI2devNBz1zQrb", "name": "Brian", "style": "tech-authority"},
    "ai-news": {"voice_id": "Yko7PKs96k2ssGkDz4Mc", "name": "Henry", "style": "forward-analyst"},
    "default": {"voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew", "style": "neutral"},
}

DEFAULT_MODEL = "eleven_turbo_v2_5"


def generate_speech(
    text: str,
    niche: str = "default",
    output_path: Optional[str] = None,
    model_id: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate speech from text. Returns MP3 bytes."""
    import requests

    key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise ValueError("ELEVENLABS_API_KEY not set")

    profile = VOICE_PROFILES.get(niche, VOICE_PROFILES["default"])
    voice_id = profile["voice_id"]

    resp = requests.post(
        f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=60,
    )
    resp.raise_for_status()
    audio_bytes = resp.content

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(audio_bytes)
        log.info(f"Audio saved to {output_path} ({len(audio_bytes)} bytes)")

    return audio_bytes


def get_voice_for_niche(niche: str) -> Dict:
    """Get the recommended voice profile for a niche."""
    return VOICE_PROFILES.get(niche, VOICE_PROFILES["default"])
