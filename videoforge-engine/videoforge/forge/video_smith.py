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
from ..knowledge.domain_expertise import get_domain_expertise
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

# Narration templates per scene role — written for TTS, not text
_NARRATION_TEMPLATES = {
    "hook": [
        "{hook_text}",
    ],
    "context": [
        "So, {topic}. Most people get this wrong.",
        "{topic}. There's a reason you keep hearing about it.",
        "Okay. {topic}. Stick with me for thirty seconds.",
        "You've been doing {topic} the hard way. Here's the easy way.",
        "Nobody taught you {topic} properly. That ends now.",
        "I tested every approach to {topic}. Only one worked.",
        "One thing about {topic} that nobody mentions.",
        "Forget what you've heard about {topic}.",
    ],
    "point_1": [
        "Start here. {point}.",
        "Number one -- {point}.",
        "First. {point}.",
        "This is the foundation. {point}.",
        "Before anything else -- {point}.",
        "Most people skip this. {point}.",
        "Dead simple. {point}.",
        "The starting point? {point}.",
    ],
    "point_2": [
        "But here's where it gets real -- {point}.",
        "Second. {point}.",
        "Now pay attention. {point}.",
        "This one catches people off guard. {point}.",
        "And it goes deeper. {point}.",
        "Here's what they don't tell you. {point}.",
        "Okay, this one's important. {point}.",
        "Most people stop at number one. Don't. {point}.",
    ],
    "point_3": [
        "And the big one. {point}.",
        "This is the one that matters. {point}.",
        "Save this. {point}.",
        "The part nobody talks about? {point}.",
        "Third, and this is the real one -- {point}.",
        "Last one. And it's the best. {point}.",
        "Here's where it all clicks. {point}.",
        "The one you'll actually remember. {point}.",
    ],
    "climax": [
        "That's it. That's the whole thing.",
        "And now you know what ninety percent of people don't.",
        "See the difference? That's not a small thing.",
        "That's the real answer. Everything else is noise.",
        "Once you see this, you can't unsee it.",
        "And that? That's what separates good from great.",
        "Read that back. Let it sink in.",
        "Simpler than you thought. But it works.",
    ],
    "cta": [
        "{cta_text}",
    ],
    "intro": [
        "Okay. {topic}. From scratch.",
        "Full breakdown of {topic}. No fluff.",
        "By the end of this, {topic} makes total sense.",
        "{topic}, explained in under a minute.",
    ],
    "section_1": [
        "The foundation first. {point}.",
        "Basics. {point}.",
        "Before anything else -- {point}.",
        "Start here. {point}.",
    ],
    "section_2": [
        "Now it gets interesting. {point}.",
        "This is the part people skip. {point}.",
        "Deeper. {point}.",
        "Here's where it actually matters. {point}.",
    ],
    "section_3": [
        "And the final piece. {point}.",
        "This connects everything. {point}.",
        "One last thing. {point}.",
        "The part that ties it together. {point}.",
    ],
    "transition_hook": [
        "But wait.",
        "And it gets better.",
        "Hold on -- this next part matters.",
        "That's not even the best part.",
    ],
    "transition_hook_2": [
        "Stay with me.",
        "Almost there.",
        "Don't skip this.",
        "The best part's coming.",
    ],
}

# Niche-specific point generators -- replace generic "{point}" with niche-aware content
_NICHE_POINT_TEMPLATES = {
    "witchcraft": [
        "your intention is the engine. the tools just focus it",
        "one white candle and five minutes of focus. that's all you need to start",
        "the herb, the crystal, and the intention -- they work together, not alone",
        "timing it with the moon isn't optional. it's the difference between hoping and doing",
        "your practice is yours. nobody else's {topic} will look like yours",
        "forget buying more crystals. learn to use the three you already have",
        "the oldest spells are the simplest. that's not a coincidence",
        "write it down. spoken intention is good. written intention is permanent",
    ],
    "mythology": [
        "this story is over three thousand years old and we're still telling it wrong",
        "the myth wasn't entertainment. it was a survival manual",
        "every culture has this same story. different names. same warning",
        "the gods weren't perfect. that was the whole point",
        "the symbolism goes layers deep. the surface story is a distraction",
        "civilizations built empires around this belief. then collapsed when they forgot it",
        "the real version is darker, stranger, and more interesting than the one you know",
        "this character shows up in Greek, Norse, and Egyptian myth. same role every time",
    ],
    "tech": [
        "five minute setup. saves five hours a week. do the math",
        "this one setting changes everything. most people never find it",
        "stop overcomplicating it. one device, one app, done",
        "the automation is the product. the hardware is just the trigger",
        "check compatibility first. this mistake costs people hundreds of dollars",
        "the thirty dollar version outperforms the hundred dollar one. I tested both",
        "one routine replaces six manual steps. set it once, forget it",
        "everyone buys this. almost nobody sets it up right",
    ],
    "ai_news": [
        "this dropped yesterday and it changes how three industries work",
        "the headlines got it wrong. here's what actually happened",
        "this tool went from zero to a million users in four days. there's a reason",
        "your job description just changed. whether you noticed or not",
        "the big companies adopted this quietly. now you can too",
        "open source just caught up to the closed models. that's a big deal",
        "this benchmark score matters. here's why in plain english",
        "three months ago this was impossible. now it takes ten seconds",
    ],
    "lifestyle": [
        "two minutes in the morning. that's the whole habit",
        "consistency beats intensity. every single time",
        "build the system first. motivation comes and goes",
        "the people who stick with this all do one thing differently",
        "start with the smallest version that still counts",
        "this costs nothing and works better than the expensive option",
        "track it for one week. the data will surprise you",
        "one less decision in the morning changes the whole day",
    ],
    "fitness": [
        "wore this for thirty days straight. here's the real accuracy",
        "the heart rate sensor matters more than the step counter",
        "battery life claims versus reality. not even close",
        "the cheaper model tracks everything the premium does. minus one feature",
        "comfort after eight hours -- that's the real test",
        "the app makes or breaks the device. hardware alone means nothing",
        "waterproof rating versus actual pool use. big difference",
        "GPS accuracy on trails, not just sidewalks. that's the real benchmark",
    ],
    "business": [
        "this tool paid for itself in the first week",
        "automate the boring part. spend your time on the part that makes money",
        "four hundred dollars in the first month. exact steps, no fluff",
        "the free tier does ninety percent of what the paid version does",
        "one prompt, three outputs, five platforms. ten minutes total",
        "stop trading hours for dollars. this is the setup that broke that cycle",
        "the ROI isn't theoretical. I tracked every dollar",
        "most people use ten percent of this tool. here's the other ninety",
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

        Each scene's duration = max(min_dur, (word_count / wpm) * 60 + 0.2)
        capped at max_dur. The +0.2s buffer gives minimal breathing room.
        Actual audio duration overrides this in the render engine.
        """
        wpm = pacing.get("word_rate_wpm", 160)
        min_dur = pacing.get("min_scene_duration", 1.0)
        max_dur = pacing.get("max_scene_duration", 8.0)

        for scene in scenes:
            word_count = len(scene.narration.split()) if scene.narration else 0
            if word_count > 0:
                speech_dur = (word_count / wpm) * 60 + 0.2
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
        """Generate a visual description prompt using domain expertise visual subjects.

        Uses specific visual_subjects from domain_expertise.py matched to the topic,
        falling back to niche defaults, then visual_dna key_visuals.
        """
        # Get domain expertise visual subjects for this niche
        expertise = get_domain_expertise(niche, topic)
        visual_subjects = expertise.get("visual_subjects", {})

        # Pick the best visual subject: topic-matched > default > fallback
        matched = expertise.get("matched_visual", {})
        subject = matched.get("description", "") if matched else ""
        if not subject:
            subject = visual_subjects.get("default", "")

        # Fallback to visual_dna key_visuals if no expertise
        if not subject:
            visual_dna = profile.get("visual_dna", {})
            key_visuals = visual_dna.get("key_visuals", [])
            subject = ", ".join(key_visuals[:3]) if key_visuals else topic

        # Role-specific composition direction (what the camera does)
        compositions = {
            "hook": f"Dramatic hero shot, {subject}, bold composition, intense lighting, cinematic wide angle",
            "context": f"Establishing shot, {subject}, atmospheric wide view",
            "point_1": f"Close-up detail, {subject}, sharp focus, informative framing",
            "point_2": f"Overhead flat lay view, {subject}, organized composition, clean layout",
            "point_3": f"Dynamic angle, {subject}, engaging composition, dramatic perspective",
            "climax": f"Epic shot, {subject}, dramatic lighting, pinnacle moment",
            "cta": f"Cinematic wide shot, {subject}, space for text overlay, clean lower third",
            "intro": f"Wide establishing scene, {subject}, atmospheric opening",
            "section_1": f"Medium shot, {subject}, informative framing",
            "section_2": f"Dynamic composition, {subject}, engaging detail",
            "section_3": f"B-roll variety shot, {subject}, visual interest",
            "transition_hook": f"Dramatic angle, {subject}, bold visual, high energy",
            "transition_hook_2": f"Extreme close-up, {subject}, texture detail, macro feel",
        }
        return compositions.get(role, f"{subject}, professional composition")

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
