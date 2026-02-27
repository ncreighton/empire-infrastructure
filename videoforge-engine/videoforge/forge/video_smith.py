"""VideoSmith — Template-based storyboard and audio plan generation (zero AI cost)."""

import re
from datetime import datetime, timezone
from ..models import (
    SceneSpec, Storyboard, AudioPlan, SubtitleTrack, VideoPlan,
)
from ..knowledge.niche_profiles import get_niche_profile
from ..knowledge.hook_formulas import HOOK_FORMULAS, get_best_hook
from ..knowledge.platform_specs import get_platform_spec
from ..knowledge.pacing import get_pacing
from ..knowledge.music_moods import get_mood_for_niche
from ..knowledge.color_grades import get_color_grade
from ..knowledge.subtitle_styles import get_subtitle_style
from ..knowledge.shot_types import SHOT_TYPES
from ..knowledge.transitions import TRANSITIONS
from ..voice import get_voice, get_elevenlabs_voice
from .variation_engine import (
    VariationEngine, HOOK_OPENING_POOLS, CTA_POOLS,
    TRANSITION_PHRASE_POOLS, THUMBNAIL_CONCEPT_POOLS,
)


# Storyboard templates by format
_SHORT_TEMPLATE = [
    {"role": "hook", "duration": 2.0, "shot": "slow_zoom_in", "transition": "cut"},
    {"role": "context", "duration": 4.0, "shot": "medium", "transition": "slide_left"},
    {"role": "point_1", "duration": 5.0, "shot": "close_up", "transition": "cut"},
    {"role": "point_2", "duration": 5.0, "shot": "overhead_flat_lay", "transition": "crossfade"},
    {"role": "point_3", "duration": 5.0, "shot": "slow_zoom_in", "transition": "cut"},
    {"role": "climax", "duration": 4.0, "shot": "low_angle", "transition": "flash"},
    {"role": "cta", "duration": 3.0, "shot": "static_locked", "transition": "fade_black"},
]

_STANDARD_TEMPLATE = [
    {"role": "hook", "duration": 5.0, "shot": "wide_establishing", "transition": "cut"},
    {"role": "intro", "duration": 10.0, "shot": "medium", "transition": "crossfade"},
    {"role": "section_1", "duration": 25.0, "shot": "close_up", "transition": "slide_left"},
    {"role": "transition_hook", "duration": 3.0, "shot": "dutch_angle", "transition": "flash"},
    {"role": "section_2", "duration": 25.0, "shot": "tracking_forward", "transition": "cut"},
    {"role": "transition_hook_2", "duration": 3.0, "shot": "extreme_close_up", "transition": "whip_pan"},
    {"role": "section_3", "duration": 25.0, "shot": "b_roll_montage", "transition": "crossfade"},
    {"role": "climax", "duration": 10.0, "shot": "slow_zoom_in", "transition": "fade_black"},
    {"role": "cta", "duration": 10.0, "shot": "static_locked", "transition": "crossfade"},
]

# Narration templates per scene role (expanded for variation)
_NARRATION_TEMPLATES = {
    "hook": [
        "{hook_text}",
    ],
    "context": [
        "Here's what you need to know about {topic}.",
        "Let's talk about {topic} — and why it matters right now.",
        "Today we're diving into {topic}.",
        "{topic} is about to change how you think about this.",
        "Most people get {topic} completely wrong. Here's the truth.",
        "I've spent years studying {topic}. Here's what I found.",
        "If you only learn one thing about {topic}, make it this.",
        "Everything you thought you knew about {topic}? Forget it.",
    ],
    "point_1": [
        "First up: {point}.",
        "Number one — {point}.",
        "The first thing you should know: {point}.",
        "Let's start here. {point}.",
        "Right off the bat — {point}.",
        "This is foundational. {point}.",
        "Most people miss this. {point}.",
        "Here's where it begins. {point}.",
    ],
    "point_2": [
        "But here's the thing — {point}.",
        "Number two. {point}.",
        "Here's another key insight: {point}.",
        "Now it gets interesting. {point}.",
        "This one surprises everyone. {point}.",
        "Pay attention to this. {point}.",
        "And it doesn't stop there. {point}.",
        "Here's what most people overlook. {point}.",
    ],
    "point_3": [
        "And this is the one that matters most. {point}.",
        "Last but not least — {point}.",
        "The biggest takeaway: {point}.",
        "Save this one. {point}.",
        "This changes everything. {point}.",
        "And the final piece: {point}.",
        "Here's the part nobody talks about. {point}.",
        "The most important one: {point}.",
    ],
    "climax": [
        "This is the game-changer.",
        "And here's the part that changes everything.",
        "Now you see the full picture.",
        "That's the real secret. And now you know it.",
        "This is what separates the beginners from the experts.",
        "And that? That changes everything about {topic}.",
        "Once you understand this, you'll never go back.",
        "This is the moment it all clicks.",
    ],
    "cta": [
        "{cta_text}",
    ],
    "intro": [
        "Let's break down {topic} — from the ground up.",
        "Today we're going deep on {topic}.",
        "By the end of this, you'll understand {topic} completely.",
        "This is the complete breakdown of {topic}.",
    ],
    "section_1": [
        "Let's start with the foundation. {point}.",
        "The basics first. {point}.",
        "Before we go further — {point}.",
        "Ground zero. {point}.",
    ],
    "section_2": [
        "Now here's where it gets interesting. {point}.",
        "This is the part most people skip. {point}.",
        "Let's go deeper. {point}.",
        "But here's what changes the game. {point}.",
    ],
    "section_3": [
        "And the final piece of the puzzle. {point}.",
        "This ties everything together. {point}.",
        "And the last thing you need to know. {point}.",
        "The grand finale. {point}.",
    ],
    "transition_hook": [
        "But wait — there's more.",
        "And it gets better.",
        "Hold on. This next part is key.",
        "But that's not even the best part.",
    ],
    "transition_hook_2": [
        "Stay with me — the best part is coming.",
        "We're just getting started.",
        "Don't skip this next part.",
        "Now watch what happens next.",
    ],
}

# Niche-specific point generators — replace generic "{point}" with niche-aware content
_NICHE_POINT_TEMPLATES = {
    "witchcraft": [
        "the power of intention is real, and {topic} proves it",
        "your energy and focus are the most powerful tools you have",
        "the correspondences between herbs, crystals, and intention matter",
        "timing your practice with the moon amplifies everything",
        "building a personal practice changes your entire relationship with {topic}",
    ],
    "mythology": [
        "the ancient stories carry wisdom we're only now rediscovering",
        "this myth reveals something fundamental about human nature",
        "the symbolism goes deeper than most scholars realize",
        "civilizations rose and fell based on these beliefs",
        "the parallels between cultures tell us something profound about {topic}",
    ],
    "tech": [
        "the setup takes five minutes but saves you hours every week",
        "this one integration changes how your entire system works",
        "most people overcomplicate this — keep it simple",
        "the real power is in the automation, not the device",
        "compatibility is everything — check this before you buy",
    ],
    "ai_news": [
        "this AI breakthrough just changed everything in the industry",
        "the real impact isn't what the headlines are telling you",
        "this technology is already being used by top companies",
        "the implications for your career are massive",
        "most people won't notice this shift until it's too late",
    ],
    "lifestyle": [
        "this small change makes a bigger difference than you'd expect",
        "consistency matters more than perfection",
        "it's about building systems, not relying on motivation",
        "the people who succeed at this all have one thing in common",
        "start here — it takes less than ten minutes",
    ],
}


class VideoSmith:
    """Template-based storyboard generator with anti-repetition."""

    def __init__(self, db_path: str = None):
        self.variation = VariationEngine(db_path=db_path)

    def craft_storyboard(self, topic: str, niche: str,
                         platform: str = "youtube_shorts",
                         format: str = "short",
                         hook_formula: str = None) -> Storyboard:
        """Generate a complete storyboard from templates."""
        profile = get_niche_profile(niche)
        pacing = get_pacing(platform=platform, niche=niche)
        spec = get_platform_spec(platform)
        voice = get_voice(niche)
        color = get_color_grade(niche=niche)
        moods = get_mood_for_niche(niche)
        category = profile.get("category", "tech")

        # Select hook formula
        if not hook_formula:
            hook_formula = get_best_hook(niche)
        formula = HOOK_FORMULAS.get(hook_formula, {})

        # Generate hook text
        hook_text = self._generate_hook(topic, niche, hook_formula, formula)

        # Generate CTA
        cta_text = self._generate_cta(niche, platform)

        # Select template
        template = _SHORT_TEMPLATE if format == "short" else _STANDARD_TEMPLATE

        # Build scenes
        scenes = []
        for i, tpl in enumerate(template):
            narration = self._generate_narration(
                tpl["role"], topic, hook_text, cta_text, niche
            )
            visual_prompt = self._generate_visual_prompt(
                tpl["role"], topic, niche, profile
            )

            scenes.append(SceneSpec(
                scene_number=i + 1,
                duration_seconds=tpl["duration"],
                narration=narration,
                visual_prompt=visual_prompt,
                shot_type=tpl["shot"],
                transition_in=tpl["transition"],
                text_overlay=self._get_text_overlay(tpl["role"], hook_text, cta_text),
                music_cue="" if i > 0 else "start",
                subtitle_text=narration,
            ))

        # Voice-driven scene durations based on narration word count
        self._calculate_scene_durations(scenes, pacing)

        # Thumbnail concept
        thumb_pool = THUMBNAIL_CONCEPT_POOLS.get(category, THUMBNAIL_CONCEPT_POOLS.get("tech", []))
        thumbnail = self.variation.pick(f"thumbnail_{niche}", thumb_pool) if thumb_pool else ""

        # Hashtags from niche profile
        hashtags = profile.get("hashtags", [])[:spec.get("max_hashtags", 15)]

        # Subtitle style
        sub_style = "hormozi" if format == "short" else "ali_abdaal"

        return Storyboard(
            title=self._generate_title(topic),
            niche=niche,
            platform=platform,
            format=format,
            total_duration=sum(s.duration_seconds for s in scenes),
            scenes=scenes,
            hook_formula=hook_formula,
            cta_text=cta_text,
            thumbnail_concept=thumbnail,
            hashtags=hashtags,
            music_mood=moods[0] if moods else "lo_fi",
            subtitle_style=sub_style,
            color_grade=color.get("name", ""),
            voice_id=voice.get("voice_id", ""),
        )

    def craft_audio_plan(self, storyboard: Storyboard, niche: str) -> AudioPlan:
        """Generate an audio plan from a storyboard."""
        voice = get_voice(niche)
        el_voice = get_elevenlabs_voice(niche)
        moods = get_mood_for_niche(niche)

        return AudioPlan(
            voice_id=voice["voice_id"],
            voice_name=voice["name"],
            tts_provider="elevenlabs",
            music_track=moods[0] if moods else "lo_fi",
            music_source="pixabay",
            music_volume=0.15,
            sfx_cues=["whoosh", "pop"],
            elevenlabs_voice_id=el_voice["voice_id"],
            elevenlabs_model="eleven_turbo_v2_5",
            voice_settings={
                "stability": el_voice["stability"],
                "similarity_boost": el_voice["similarity_boost"],
                "style": el_voice["style"],
            },
        )

    def craft_subtitle_track(self, storyboard: Storyboard) -> SubtitleTrack:
        """Generate a subtitle track from storyboard narration."""
        style_data = get_subtitle_style(storyboard.subtitle_style)
        segments = []
        current_time = 0.0

        for scene in storyboard.scenes:
            if scene.narration:
                segments.append({
                    "start": round(current_time, 2),
                    "end": round(current_time + scene.duration_seconds, 2),
                    "text": scene.narration,
                    "highlight": "",
                })
            current_time += scene.duration_seconds

        return SubtitleTrack(
            style=storyboard.subtitle_style,
            segments=segments,
            font=style_data.get("font", "Montserrat"),
            font_size=style_data.get("font_size", 48),
            color=style_data.get("color", "#FFFFFF"),
            highlight_color=style_data.get("highlight_color", "#FFD700"),
            background=style_data.get("background", ""),
            position=style_data.get("position", "center"),
        )

    def to_video_plan(self, topic: str, niche: str,
                      platform: str = "youtube_shorts",
                      format: str = "short") -> VideoPlan:
        """Generate a complete VideoPlan with storyboard, audio, and subtitles."""
        sb = self.craft_storyboard(topic, niche, platform, format)
        audio = self.craft_audio_plan(sb, niche)
        subs = self.craft_subtitle_track(sb)

        return VideoPlan(
            topic=topic,
            niche=niche,
            platform=platform,
            format=format,
            storyboard=sb,
            audio_plan=audio,
            subtitle_track=subs,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="draft",
        )

    # ── Duration & Scene Calculation ─────────────────────────────────

    def _calculate_scene_durations(self, scenes: list, pacing: dict):
        """Calculate voice-driven scene durations based on narration word count.

        Each scene's duration = max(min_dur, (word_count / wpm) * 60 + 1.0)
        capped at max_dur. Light scaling only if total deviates >20% from target.
        """
        wpm = pacing.get("word_rate_wpm", 160)
        min_dur = pacing.get("min_scene_duration", 1.0)
        max_dur = pacing.get("max_scene_duration", 5.0)

        for scene in scenes:
            word_count = len(scene.narration.split()) if scene.narration else 0
            if word_count > 0:
                speech_dur = (word_count / wpm) * 60 + 1.0
                scene.duration_seconds = round(
                    max(min_dur, min(speech_dur, max_dur)), 1
                )
            else:
                scene.duration_seconds = round(min_dur, 1)

        # Light scaling only if total deviates >20% from target
        ideal_duration = pacing.get("ideal_total_duration", (30, 60))
        target = (ideal_duration[0] + ideal_duration[1]) / 2
        current_total = sum(s.duration_seconds for s in scenes)
        if current_total > 0:
            ratio = target / current_total
            if ratio < 0.8 or ratio > 1.2:
                for s in scenes:
                    s.duration_seconds = round(s.duration_seconds * ratio, 1)

    # ── Generators ────────────────────────────────────────────────────

    def _generate_hook(self, topic: str, niche: str, formula_key: str, formula: dict) -> str:
        """Generate hook text using formula templates + variation engine."""
        pool = HOOK_OPENING_POOLS.get(formula_key, HOOK_OPENING_POOLS.get("curiosity_gap", []))
        if pool:
            hook = self.variation.pick(f"hook_{niche}_{formula_key}", pool)
            hook = hook.replace("{topic}", topic).replace("{n}", "5").replace("{N}", "5")
            return hook
        # Fallback to formula templates
        templates = formula.get("templates", [])
        if templates:
            tpl = self.variation.pick(f"hook_tpl_{niche}", templates)
            return tpl.replace("{topic}", topic)
        return f"Let's talk about {topic}."

    def _generate_cta(self, niche: str, platform: str) -> str:
        """Generate CTA text."""
        pool_key = "subscribe" if platform == "youtube" else "follow"
        pool = CTA_POOLS.get(pool_key, CTA_POOLS["engagement"])
        profile = get_niche_profile(niche)
        brand = profile.get("brand", niche)

        cta = self.variation.pick(f"cta_{niche}", pool)
        cta = cta.replace("{niche}", brand.lower())
        return cta

    def _generate_narration(self, role: str, topic: str, hook_text: str,
                            cta_text: str, niche: str) -> str:
        """Generate narration for a scene based on role."""
        templates = _NARRATION_TEMPLATES.get(role, [f"More about {topic}."])
        text = self.variation.pick(f"narration_{role}_{niche}", templates)
        text = text.replace("{topic}", topic)
        text = text.replace("{hook_text}", hook_text)
        text = text.replace("{cta_text}", cta_text)

        # Replace {point} with niche-aware content instead of generic filler
        if "{point}" in text:
            profile = get_niche_profile(niche)
            category = profile.get("category", "tech")
            point_pool = _NICHE_POINT_TEMPLATES.get(
                category, _NICHE_POINT_TEMPLATES.get("tech", [])
            )
            if point_pool:
                point = self.variation.pick(f"point_{role}_{niche}", point_pool)
                point = point.replace("{topic}", topic)
            else:
                point = f"this key aspect of {topic} that most people miss"
            text = text.replace("{point}", point)

        return text

    def _generate_visual_prompt(self, role: str, topic: str,
                                niche: str, profile: dict) -> str:
        """Generate a visual description prompt for AI image generation."""
        visual_dna = profile.get("visual_dna", {})
        aesthetic = visual_dna.get("aesthetic", "clean")
        key_visuals = visual_dna.get("key_visuals", [])
        palette = visual_dna.get("color_palette", [])

        visuals_str = ", ".join(key_visuals[:3]) if key_visuals else topic
        colors_str = " and ".join(palette[:2]) if palette else ""

        prompts = {
            "hook": f"Dramatic hero shot, {aesthetic} style, {topic}, {visuals_str}, bold composition, intense lighting, {colors_str} color scheme, cinematic wide angle",
            "context": f"{aesthetic} establishing shot, {topic} environment, {visuals_str}, atmospheric",
            "point_1": f"Close-up detail shot, {topic} related, {visuals_str}, {aesthetic} style",
            "point_2": f"Overhead view, {topic} arrangement, {visuals_str}, organized composition",
            "point_3": f"Dynamic angle, {topic} highlight, {visuals_str}, engaging composition",
            "climax": f"Epic {aesthetic} shot, {topic} pinnacle moment, {visuals_str}, dramatic lighting",
            "cta": f"Cinematic wide shot, {aesthetic} style, {topic}, {visuals_str}, {colors_str}, space for text overlay, clean lower third",
            "intro": f"Wide {aesthetic} scene, {topic} world, {visuals_str}, establishing atmosphere",
            "section_1": f"Medium shot, {topic} focus, {visuals_str}, {aesthetic} style, informative",
            "section_2": f"Dynamic composition, {topic} deep dive, {visuals_str}, {aesthetic}",
            "section_3": f"B-roll montage style, {topic} variety, {visuals_str}, {aesthetic}",
            "transition_hook": f"Dramatic angle, {aesthetic} style, {topic}, {visuals_str}, {colors_str}, bold visual",
            "transition_hook_2": f"Extreme close-up, {aesthetic} texture, {topic}, {visuals_str}, {colors_str}, detail shot",
        }
        return prompts.get(role, f"{aesthetic} style, {topic}, {visuals_str}")

    def _get_text_overlay(self, role: str, hook_text: str, cta_text: str) -> str:
        """Get text overlay for specific scene roles."""
        if role == "hook":
            # First few impactful words
            words = hook_text.split()[:5]
            return " ".join(words).upper()
        if role == "cta":
            return cta_text
        if role in ("transition_hook", "transition_hook_2"):
            return "But wait..."
        return ""

    def _generate_title(self, topic: str) -> str:
        """Generate a video title from the topic."""
        # Clean up and capitalize
        title = topic.strip()
        if not title[0].isupper():
            title = title.capitalize()
        # Truncate for platform limits
        if len(title) > 90:
            title = title[:87] + "..."
        return title
