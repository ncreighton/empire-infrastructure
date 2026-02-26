"""AudioEngine — Edge TTS for narration + free music sourcing."""

import os
import logging
import asyncio
import tempfile
from ..models import AudioPlan, Storyboard
from ..voice import get_voice

logger = logging.getLogger(__name__)


class AudioEngine:
    """Generates audio assets: TTS narration and background music."""

    def generate_narration(self, text: str, voice_id: str,
                           rate: str = "0%", pitch: str = "0st",
                           output_path: str = None) -> str:
        """Generate TTS narration using edge-tts.

        Returns path to generated audio file.
        """
        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), "videoforge_narration.mp3")

        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return ""

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
            logger.error(f"TTS generation failed: {e}")
            return ""

    def generate_full_narration(self, storyboard: Storyboard,
                                niche: str,
                                output_dir: str = None) -> list:
        """Generate TTS audio for each scene in a storyboard.

        Returns list of {scene_number, path, text} dicts.
        """
        if not output_dir:
            output_dir = os.path.join(tempfile.gettempdir(), "videoforge_audio")
        os.makedirs(output_dir, exist_ok=True)

        voice = get_voice(niche)
        results = []

        for scene in storyboard.scenes:
            if not scene.narration:
                continue

            output_path = os.path.join(output_dir, f"scene_{scene.scene_number}.mp3")
            path = self.generate_narration(
                text=scene.narration,
                voice_id=voice["voice_id"],
                rate=voice.get("rate", "0%"),
                pitch=voice.get("pitch", "0st"),
                output_path=output_path,
            )

            results.append({
                "scene_number": scene.scene_number,
                "path": path,
                "text": scene.narration,
                "voice_id": voice["voice_id"],
            })

        return results

    def get_music_recommendation(self, audio_plan: AudioPlan) -> dict:
        """Get music recommendation with search terms for sourcing.

        Note: Actual music download/licensing handled externally.
        Returns recommendation dict with source, search_terms, volume.
        """
        from ..knowledge.music_moods import MUSIC_MOODS
        from ..knowledge.audio_library import get_music_source

        mood_key = audio_plan.music_track
        mood_data = MUSIC_MOODS.get(mood_key, {})
        source_data = get_music_source(mood_key)

        return {
            "mood": mood_key,
            "mood_name": mood_data.get("name", mood_key),
            "energy": mood_data.get("energy", "medium"),
            "bpm_range": mood_data.get("bpm_range", (90, 120)),
            "source": source_data.get("source", "pixabay_music"),
            "search_terms": source_data.get("search_terms", ["background", "ambient"]),
            "keywords": mood_data.get("keywords", []),
            "volume": audio_plan.music_volume,
        }

    def estimate_tts_duration(self, text: str, wpm: int = 150) -> float:
        """Estimate TTS duration in seconds from text."""
        word_count = len(text.split())
        return (word_count / wpm) * 60
