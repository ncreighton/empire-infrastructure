#!/usr/bin/env python3
"""
ForgeFiles ElevenLabs TTS Engine
===================================
Generates AI voiceover audio via the ElevenLabs text-to-speech API.
Supports voice selection, model choice, and automatic upload to
catbox.moe for Creatomate-accessible public URLs.

Usage:
    python elevenlabs_tts.py --text "Check out this Dragon Guardian." --output voiceover.mp3
    python elevenlabs_tts.py --test
    python elevenlabs_tts.py --list-voices
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

# Add scripts dir to path for local imports
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

CONFIG_PATH = SCRIPTS_DIR.parent / "config" / "pipeline_config.json"

# ElevenLabs API configuration
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Default voice: George — deep, warm, confident (ideal for product showcase)
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_MODEL = "eleven_multilingual_v2"

# Voice settings tuned for product narration
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.2,
    "use_speaker_boost": True,
}

# Catbox.moe for temporary public file hosting (Creatomate needs public URLs)
CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    """Load ElevenLabs config from pipeline_config.json."""
    config = {
        "api_key": os.environ.get("ELEVENLABS_API_KEY", ""),
        "voice_id": DEFAULT_VOICE_ID,
        "model": DEFAULT_MODEL,
        "voice_settings": DEFAULT_VOICE_SETTINGS.copy(),
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            if cfg.get("elevenlabs_api_key"):
                config["api_key"] = cfg["elevenlabs_api_key"]
            if cfg.get("elevenlabs_voice_id"):
                config["voice_id"] = cfg["elevenlabs_voice_id"]
            if cfg.get("elevenlabs_model"):
                config["model"] = cfg["elevenlabs_model"]
        except (json.JSONDecodeError, IOError):
            pass

    # Environment variable overrides config file
    env_key = os.environ.get("ELEVENLABS_API_KEY")
    if env_key:
        config["api_key"] = env_key

    return config


# ============================================================================
# API HELPERS
# ============================================================================

def _api_request(endpoint, method="GET", data=None, api_key=None, raw_response=False):
    """Make a request to the ElevenLabs API."""
    url = f"{ELEVENLABS_API_BASE}{endpoint}"
    headers = {
        "xi-api-key": api_key,
    }

    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")
    else:
        body = None

    req = Request(url, data=body, headers=headers, method=method)

    try:
        response = urlopen(req, timeout=120)
        if raw_response:
            return response.read()
        return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"[ElevenLabs] API error {e.code}: {error_body}")
        return None
    except URLError as e:
        print(f"[ElevenLabs] Connection error: {e.reason}")
        return None


# ============================================================================
# VOICE MANAGEMENT
# ============================================================================

def list_voices(api_key=None):
    """List available ElevenLabs voices."""
    config = load_config()
    api_key = api_key or config["api_key"]

    if not api_key:
        print("[ElevenLabs] ERROR: No API key configured")
        return None

    result = _api_request("/voices", api_key=api_key)
    if result and "voices" in result:
        return result["voices"]
    return None


def get_voice_info(voice_id=None, api_key=None):
    """Get details about a specific voice."""
    config = load_config()
    voice_id = voice_id or config["voice_id"]
    api_key = api_key or config["api_key"]

    if not api_key:
        return None

    return _api_request(f"/voices/{voice_id}", api_key=api_key)


# ============================================================================
# TEXT-TO-SPEECH
# ============================================================================

def generate_voiceover(text, output_path, voice_id=None, model=None,
                       voice_settings=None, api_key=None):
    """Generate voiceover MP3 via ElevenLabs API.

    Args:
        text: The narration text to synthesize
        output_path: Where to save the MP3 file
        voice_id: ElevenLabs voice ID (default: George)
        model: TTS model ID (default: eleven_multilingual_v2)
        voice_settings: Dict with stability, similarity_boost, style
        api_key: API key (falls back to config/env)

    Returns:
        Path to the generated MP3 file, or None on failure
    """
    config = load_config()
    api_key = api_key or config["api_key"]
    voice_id = voice_id or config["voice_id"]
    model = model or config["model"]
    voice_settings = voice_settings or config["voice_settings"]

    if not api_key:
        print("[ElevenLabs] ERROR: No API key. Set ELEVENLABS_API_KEY or add to pipeline_config.json")
        return None

    if not text or not text.strip():
        print("[ElevenLabs] ERROR: Empty text provided")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": voice_settings,
    }

    print(f"[ElevenLabs] Generating voiceover ({len(text)} chars, voice={voice_id[:8]}...)")
    start_time = time.time()

    audio_data = _api_request(
        f"/text-to-speech/{voice_id}",
        method="POST",
        data=payload,
        api_key=api_key,
        raw_response=True,
    )

    if audio_data is None:
        print("[ElevenLabs] ERROR: TTS generation failed")
        return None

    with open(output_path, "wb") as f:
        f.write(audio_data)

    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size
    print(f"[ElevenLabs] Voiceover saved: {output_path} ({file_size / 1024:.1f} KB, {elapsed:.1f}s)")

    return str(output_path)


def generate_voiceover_with_timestamps(text, output_path, voice_id=None,
                                        model=None, api_key=None):
    """Generate voiceover with word-level timestamps for subtitle sync.

    Uses the /text-to-speech/{voice_id}/with-timestamps endpoint.
    Returns (audio_path, timestamps) tuple.
    """
    config = load_config()
    api_key = api_key or config["api_key"]
    voice_id = voice_id or config["voice_id"]
    model = model or config["model"]

    if not api_key:
        print("[ElevenLabs] ERROR: No API key configured")
        return None, None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": config["voice_settings"],
    }

    result = _api_request(
        f"/text-to-speech/{voice_id}/with-timestamps",
        method="POST",
        data=payload,
        api_key=api_key,
    )

    if result is None:
        return None, None

    # Decode base64 audio
    import base64
    audio_b64 = result.get("audio_base64", "")
    if audio_b64:
        audio_data = base64.b64decode(audio_b64)
        with open(output_path, "wb") as f:
            f.write(audio_data)

    timestamps = result.get("alignment", {})
    return str(output_path), timestamps


# ============================================================================
# FILE HOSTING (catbox.moe)
# ============================================================================

def upload_to_catbox(file_path):
    """Upload a file to catbox.moe for temporary public hosting.

    Creatomate requires public URLs for audio/video sources.
    catbox.moe provides free, no-auth temporary hosting.

    Returns:
        Public URL string, or None on failure
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[Catbox] ERROR: File not found: {file_path}")
        return None

    print(f"[Catbox] Uploading {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)...")

    # Build multipart form data manually (no requests dependency)
    boundary = "----ForgeFilesBoundary"
    body = b""

    # reqtype field
    body += f"--{boundary}\r\n".encode()
    body += b"Content-Disposition: form-data; name=\"reqtype\"\r\n\r\n"
    body += b"fileupload\r\n"

    # file field
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="fileToUpload"; filename="{file_path.name}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    with open(file_path, "rb") as f:
        body += f.read()
    body += b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    req = Request(
        CATBOX_UPLOAD_URL,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        response = urlopen(req, timeout=120)
        url = response.read().decode("utf-8").strip()
        if url.startswith("https://"):
            print(f"[Catbox] Uploaded: {url}")
            return url
        else:
            print(f"[Catbox] Unexpected response: {url}")
            return None
    except (HTTPError, URLError) as e:
        print(f"[Catbox] Upload failed: {e}")
        return None


# ============================================================================
# QUOTA CHECK
# ============================================================================

def check_quota(api_key=None):
    """Check remaining ElevenLabs character quota."""
    config = load_config()
    api_key = api_key or config["api_key"]

    if not api_key:
        return None

    result = _api_request("/user/subscription", api_key=api_key)
    if result:
        return {
            "character_count": result.get("character_count", 0),
            "character_limit": result.get("character_limit", 0),
            "remaining": result.get("character_limit", 0) - result.get("character_count", 0),
            "tier": result.get("tier", "unknown"),
        }
    return None


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ForgeFiles ElevenLabs TTS Engine")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--file", type=str, help="Read text from file")
    parser.add_argument("--output", "-o", type=str, default="voiceover.mp3",
                       help="Output MP3 path")
    parser.add_argument("--voice", type=str, default=None,
                       help="ElevenLabs voice ID")
    parser.add_argument("--model", type=str, default=None,
                       help="TTS model ID")
    parser.add_argument("--upload", action="store_true",
                       help="Upload to catbox.moe after generation")
    parser.add_argument("--list-voices", action="store_true",
                       help="List available voices")
    parser.add_argument("--quota", action="store_true",
                       help="Check character quota")
    parser.add_argument("--test", action="store_true",
                       help="Run a quick test generation")

    args = parser.parse_args()

    if args.list_voices:
        voices = list_voices()
        if voices:
            print(f"\nAvailable Voices ({len(voices)}):")
            print("=" * 70)
            for v in voices:
                labels = v.get("labels", {})
                accent = labels.get("accent", "")
                age = labels.get("age", "")
                gender = labels.get("gender", "")
                desc = labels.get("description", "")
                print(f"  {v['voice_id'][:12]}...  {v['name']:<20} "
                      f"{gender:<8} {age:<12} {accent:<12} {desc}")
        else:
            print("Failed to list voices. Check API key.")
        return

    if args.quota:
        quota = check_quota()
        if quota:
            print(f"\nElevenLabs Quota:")
            print(f"  Tier: {quota['tier']}")
            print(f"  Used: {quota['character_count']:,} / {quota['character_limit']:,}")
            print(f"  Remaining: {quota['remaining']:,} characters")
        else:
            print("Failed to check quota. Check API key.")
        return

    if args.test:
        print("[ElevenLabs] Running test generation...")
        test_text = (
            "Welcome to ForgeFiles. "
            "This is a test of the voiceover generation system. "
            "If you can hear this, the pipeline is working correctly."
        )
        result = generate_voiceover(test_text, "test_voiceover.mp3")
        if result:
            print(f"[ElevenLabs] Test PASSED: {result}")
            # Cleanup test file
            if os.path.exists("test_voiceover.mp3"):
                size = os.path.getsize("test_voiceover.mp3")
                print(f"[ElevenLabs] File size: {size / 1024:.1f} KB")
        else:
            print("[ElevenLabs] Test FAILED")
            sys.exit(1)
        return

    # Generate voiceover from text or file
    text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    if not text:
        print("ERROR: Provide --text or --file")
        parser.print_help()
        sys.exit(1)

    result = generate_voiceover(text, args.output, voice_id=args.voice, model=args.model)

    if result and args.upload:
        url = upload_to_catbox(result)
        if url:
            print(f"Public URL: {url}")


if __name__ == "__main__":
    main()
