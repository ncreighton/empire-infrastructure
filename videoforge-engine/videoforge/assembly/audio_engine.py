"""AudioEngine — ElevenLabs TTS (primary) + Edge TTS (fallback) + free music sourcing.

Audio files are uploaded to a temporary host so Creatomate can fetch them by URL.
"""

import os
from pathlib import Path
import base64
import logging
import asyncio
import tempfile
import requests
from ..models import AudioPlan, Storyboard
from ..voice import get_voice, get_elevenlabs_voice

logger = logging.getLogger(__name__)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"

# Voice-specific words-per-minute — calibrated from actual ElevenLabs output.
# Slower voices accumulate timing errors if we assume a flat 150 WPM.
_VOICE_WPM = {
    "Drew": 140,
    "Dave": 135,
    "Brian": 155,
    "Henry": 150,
    "Daniel": 148,
    "Giovanni": 138,
    "Alice": 142,
    "Adam": 160,
    "Patrick": 152,
    "Harry": 150,
    "Rachel": 145,
    "Glinda": 138,
    "Grace": 142,
    "default": 150,
}


def _get_elevenlabs_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        env_path = Path(os.path.dirname(__file__)) / ".." / ".." / "configs" / "api_keys.env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("ELEVENLABS_API_KEY="):
                        key = line.strip().split("=", 1)[1]
    return key


class AudioEngine:
    """Generates audio assets: TTS narration and background music."""

    def generate_narration(self, text: str, niche: str = "",
                           voice_id: str = "", rate: str = "0%",
                           pitch: str = "+0Hz", output_path: str = None) -> str:
        """Generate TTS narration. Tries ElevenLabs first, falls back to Edge TTS.

        Returns path to generated audio file.
        """
        if not output_path:
            output_path = Path(tempfile.gettempdir()) / "videoforge_narration.mp3"

        # Try ElevenLabs first
        el_key = _get_elevenlabs_key()
        if el_key and niche:
            result = self._generate_elevenlabs(text, niche, output_path)
            if result:
                return result

        # Fallback to Edge TTS
        return self._generate_edge_tts(text, voice_id, rate, pitch, output_path)

    def _generate_elevenlabs(self, text: str, niche: str,
                              output_path: str) -> str:
        """Generate narration using ElevenLabs API.

        Returns path to MP3 file, or empty string on failure.
        """
        api_key = _get_elevenlabs_key()
        if not api_key:
            return ""

        el_voice = get_elevenlabs_voice(niche)
        voice_id = el_voice["voice_id"]
        if not voice_id:
            return ""

        try:
            response = requests.post(
                f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}?output_format=mp3_44100_128",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": el_voice["stability"],
                        "similarity_boost": el_voice["similarity_boost"],
                        "style": el_voice["style"],
                        "use_speaker_boost": True,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"ElevenLabs TTS: {len(response.content)} bytes → {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"ElevenLabs TTS failed: {e}")
            return ""

    def _generate_edge_tts(self, text: str, voice_id: str,
                            rate: str, pitch: str,
                            output_path: str) -> str:
        """Generate narration using Edge TTS (free fallback)."""
        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return ""

        if not voice_id:
            voice_id = "en-US-JasonNeural"

        async def _generate():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(output_path)
            return output_path

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_generate())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Edge TTS generation failed: {e}")
            return ""

    def upload_to_temp_host(self, file_path: str) -> str:
        """Upload an audio file to a temporary hosting service.

        Returns a publicly accessible URL, or empty string on failure.
        Tries multiple hosts for reliability.
        """
        if not file_path or not os.path.exists(file_path):
            return ""

        # Try tmpfiles.org first — Creatomate can fetch without User-Agent
        # (catbox.moe blocks requests without UA, which Creatomate doesn't send)
        url = self._upload_tmpfiles(file_path)
        if url:
            return url

        # Fallback to catbox.moe (works for non-Creatomate consumers)
        url = self._upload_catbox(file_path)
        if url:
            return url

        logger.warning(f"All temp hosts failed for {file_path}")
        return ""

    def _upload_catbox(self, file_path: str) -> str:
        """Upload to catbox.moe (permanent hosting, direct download links)."""
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": (os.path.basename(file_path), f, "audio/mpeg")},
                    timeout=60,
                )
                response.raise_for_status()
                url = response.text.strip()
                if url.startswith("http"):
                    logger.info(f"Catbox upload: {file_path} → {url}")
                    return url
        except Exception as e:
            logger.warning(f"Catbox upload failed: {e}")
        return ""

    def _upload_tmpfiles(self, file_path: str) -> str:
        """Upload to tmpfiles.org (temporary, auto-expires)."""
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={"file": (os.path.basename(file_path), f, "audio/mpeg")},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    # tmpfiles.org returns viewing URL; convert to direct download
                    view_url = data["data"]["url"]
                    dl_url = view_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    logger.info(f"tmpfiles upload: {file_path} → {dl_url}")
                    return dl_url
        except Exception as e:
            logger.warning(f"tmpfiles upload failed: {e}")
        return ""

    def generate_full_narration(self, storyboard: Storyboard,
                                niche: str,
                                output_dir: str = None) -> list:
        """Generate TTS audio for each scene in a storyboard.

        Returns list of dicts with: scene_number, path, text, url,
        base64_data, duration_estimate, provider.

        Audio files are uploaded to a temp host so Creatomate can fetch them.
        """
        if not output_dir:
            output_dir = Path(tempfile.gettempdir()) / "videoforge_audio"
        os.makedirs(output_dir, exist_ok=True)

        voice = get_voice(niche)
        voice_name = voice.get("elevenlabs_voice_name", "")
        results = []

        for scene in storyboard.scenes:
            if not scene.narration:
                continue

            output_path = Path(output_dir) / f"scene_{scene.scene_number}.mp3"
            path = self.generate_narration(
                text=scene.narration,
                niche=niche,
                voice_id=voice["voice_id"],
                rate=voice.get("rate", "0%"),
                pitch=voice.get("pitch", "+0Hz"),
                output_path=output_path,
            )

            # Read file and base64-encode (for local preview / backup)
            base64_data = ""
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("ascii")

            # Upload to temp host for Creatomate to fetch
            audio_url = ""
            if path and os.path.exists(path):
                audio_url = self.upload_to_temp_host(path)

            # Measure actual duration from generated MP3 (most accurate),
            # falling back to voice-specific WPM estimate
            actual_duration = self.measure_mp3_duration(path) if path else 0.0
            if actual_duration <= 0:
                actual_duration = self.estimate_tts_duration(
                    scene.narration, voice_name=voice_name
                )

            results.append({
                "scene_number": scene.scene_number,
                "path": path,
                "text": scene.narration,
                "voice_id": voice["voice_id"],
                "voice_name": voice_name,
                "url": audio_url,
                "base64_data": base64_data,
                "duration_estimate": actual_duration,
                "provider": "elevenlabs" if _get_elevenlabs_key() else "edge_tts",
            })

        return results

    def get_music_recommendation(self, audio_plan: AudioPlan) -> dict:
        """Get music recommendation with search terms for sourcing."""
        from ..knowledge.music_moods import MUSIC_MOODS
        from ..knowledge.audio_library import get_music_source, get_music_url

        mood_key = audio_plan.music_track
        mood_data = MUSIC_MOODS.get(mood_key, {})
        source_data = get_music_source(mood_key)
        music_url = get_music_url(mood_key)

        return {
            "mood": mood_key,
            "mood_name": mood_data.get("name", mood_key),
            "energy": mood_data.get("energy", "medium"),
            "bpm_range": mood_data.get("bpm_range", (90, 120)),
            "source": source_data.get("source", "pixabay_music"),
            "search_terms": source_data.get("search_terms", ["background", "ambient"]),
            "keywords": mood_data.get("keywords", []),
            "volume": audio_plan.music_volume,
            "url": music_url,
        }

    def measure_mp3_duration(self, file_path: str) -> float:
        """Measure actual MP3 duration in seconds.

        Canonical: project-mesh-v2-omega/shared-core/systems/elevenlabs-tts/src/tts.py:measure_duration

        Falls back to estimation if parsing fails.
        """
        if not file_path or not os.path.exists(file_path):
            return 0.0

        try:
            # Try mutagen first (most accurate)
            from mutagen.mp3 import MP3
            audio = MP3(file_path)
            return audio.info.length
        except Exception:
            pass

        try:
            # Fallback: estimate from file size and bitrate (128kbps typical)
            file_size = os.path.getsize(file_path)
            bitrate = 128_000  # 128 kbps in bits/s
            return (file_size * 8) / bitrate
        except Exception:
            return 0.0

    def estimate_tts_duration(self, text: str, wpm: int = None,
                               voice_name: str = "") -> float:
        """Estimate TTS duration in seconds from text.

        Uses voice-specific WPM when voice_name is provided for better accuracy.
        """
        if wpm is None:
            wpm = _VOICE_WPM.get(voice_name, _VOICE_WPM["default"])
        word_count = len(text.split())
        return (word_count / wpm) * 60

    def estimate_elevenlabs_cost(self, text: str) -> float:
        """Estimate ElevenLabs cost for a text segment.

        Turbo v2.5: ~$0.00024 per character (500 chars = $0.12 per 1k chars).
        """
        return len(text) * 0.00024
