"""SubtitleEngine — Algorithmic timed subtitle generation.

Supports Hormozi, Ali Abdaal, Clean, Kinetic, and Karaoke styles.
"""

import math
from ..models import SubtitleTrack, Storyboard
from ..knowledge.subtitle_styles import SUBTITLE_STYLES, get_subtitle_style


class SubtitleEngine:
    """Generates timed subtitles for video narration."""

    def generate(self, storyboard: Storyboard,
                 style: str = None) -> SubtitleTrack:
        """Generate a complete subtitle track from storyboard narration.

        Args:
            storyboard: Storyboard with scenes containing narration
            style: Override subtitle style (default: from storyboard)

        Returns:
            SubtitleTrack with timed segments
        """
        style_key = style or storyboard.subtitle_style or "hormozi"
        style_data = get_subtitle_style(style_key)

        segments = []
        current_time = 0.0

        for scene in storyboard.scenes:
            if not scene.narration:
                current_time += scene.duration_seconds
                continue

            scene_segments = self._segment_narration(
                scene.narration,
                current_time,
                scene.duration_seconds,
                style_data,
            )
            segments.extend(scene_segments)
            current_time += scene.duration_seconds

        return SubtitleTrack(
            style=style_key,
            segments=segments,
            font=style_data.get("font", "Montserrat"),
            font_size=style_data.get("font_size", 48),
            color=style_data.get("color", "#FFFFFF"),
            highlight_color=style_data.get("highlight_color", "#FFD700"),
            background=style_data.get("background") or "",
            position=style_data.get("position", "center"),
        )

    def _segment_narration(self, text: str, start_time: float,
                           duration: float, style: dict) -> list:
        """Break narration into timed subtitle segments based on style."""
        timing_mode = style.get("timing_mode", "sentence")
        max_words = style.get("max_words_per_segment", 8)
        capitalize = style.get("capitalize", False)

        words = text.split()
        if not words:
            return []

        # Calculate timing per word
        time_per_word = duration / len(words) if words else 0

        if timing_mode == "word_by_word":
            return self._word_by_word(words, start_time, time_per_word, capitalize, style)
        elif timing_mode == "word_burst":
            return self._word_burst(words, start_time, time_per_word, max_words, capitalize, style)
        elif timing_mode == "karaoke":
            return self._karaoke(words, start_time, time_per_word, max_words, text, style)
        else:  # sentence mode
            return self._sentence_mode(words, start_time, time_per_word, max_words, capitalize, style)

    def _word_by_word(self, words: list, start: float, tpw: float,
                      capitalize: bool, style: dict) -> list:
        """Kinetic style — one word at a time."""
        segments = []
        current = start
        highlight_color = style.get("highlight_color", "#FFD700")

        for word in words:
            display = word.upper() if capitalize else word
            segments.append({
                "start": round(current, 3),
                "end": round(current + tpw, 3),
                "text": display,
                "highlight": highlight_color,
            })
            current += tpw

        return segments

    def _word_burst(self, words: list, start: float, tpw: float,
                    max_words: int, capitalize: bool, style: dict) -> list:
        """Hormozi style — 2-3 words at a time, centered, bold."""
        segments = []
        current = start
        highlight_color = style.get("highlight_color", "#FFD700")
        i = 0

        while i < len(words):
            chunk = words[i:i + max_words]
            chunk_duration = len(chunk) * tpw
            text = " ".join(chunk)
            if capitalize:
                text = text.upper()

            # Highlight the last word in the chunk for emphasis
            highlight_word = chunk[-1].upper() if capitalize else chunk[-1]

            segments.append({
                "start": round(current, 3),
                "end": round(current + chunk_duration, 3),
                "text": text,
                "highlight": highlight_word,
            })
            current += chunk_duration
            i += len(chunk)

        return segments

    def _karaoke(self, words: list, start: float, tpw: float,
                 max_words: int, full_text: str, style: dict) -> list:
        """Karaoke style — full line shown, current word highlighted."""
        segments = []
        current = start

        # Break into lines of max_words
        lines = []
        for i in range(0, len(words), max_words):
            lines.append(words[i:i + max_words])

        for line_words in lines:
            line_text = " ".join(line_words)
            line_duration = len(line_words) * tpw

            # Each word within the line gets its own highlight timing
            word_start = current
            for word in line_words:
                segments.append({
                    "start": round(word_start, 3),
                    "end": round(word_start + tpw, 3),
                    "text": line_text,
                    "highlight": word,
                })
                word_start += tpw

            current += line_duration

        return segments

    def _sentence_mode(self, words: list, start: float, tpw: float,
                       max_words: int, capitalize: bool, style: dict) -> list:
        """Clean/Ali Abdaal style — full sentences at bottom."""
        segments = []
        current = start

        # Break at sentence boundaries or max_words
        i = 0
        while i < len(words):
            chunk = []
            while i < len(words) and len(chunk) < max_words:
                chunk.append(words[i])
                i += 1
                # Break at sentence endings
                if chunk[-1].endswith((".", "!", "?")):
                    break

            chunk_duration = len(chunk) * tpw
            text = " ".join(chunk)
            if capitalize:
                text = text.upper()

            segments.append({
                "start": round(current, 3),
                "end": round(current + chunk_duration, 3),
                "text": text,
                "highlight": "",
            })
            current += chunk_duration

        return segments

    def to_srt(self, track: SubtitleTrack) -> str:
        """Export subtitle track as SRT format string."""
        lines = []
        for i, seg in enumerate(track.segments, 1):
            start = self._format_srt_time(seg["start"])
            end = self._format_srt_time(seg["end"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp HH:MM:SS,mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
