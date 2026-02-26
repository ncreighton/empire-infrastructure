"""VideoSentinel — 6-criteria quality scoring (100 points, A-F grade) with auto-enhance."""

from ..models import SentinelScore, VideoPlan, Storyboard
from ..knowledge.platform_specs import get_platform_spec
from ..knowledge.hook_formulas import HOOK_FORMULAS
from ..knowledge.subtitle_styles import SUBTITLE_STYLES
from ..knowledge.retention_patterns import RETENTION_PATTERNS


class VideoSentinel:
    """Scores video plans on 6 quality criteria and auto-enhances weak areas."""

    def score(self, plan: VideoPlan) -> SentinelScore:
        """Score a video plan across 6 criteria (100 total)."""
        result = SentinelScore()

        result.hook_strength = self._score_hook(plan)        # /20
        result.retention_arch = self._score_retention(plan)    # /20
        result.visual_quality = self._score_visual(plan)       # /15
        result.audio_quality = self._score_audio(plan)         # /15
        result.platform_opt = self._score_platform(plan)       # /15
        result.cta_effectiveness = self._score_cta(plan)       # /15

        result.calculate()
        return result

    def auto_enhance(self, plan: VideoPlan, score: SentinelScore) -> VideoPlan:
        """Patch weak areas of a video plan to raise quality score."""
        sb = plan.storyboard

        # Hook below 60% of max (12/20)
        if score.hook_strength < 12 and sb:
            score.suggestions.append("Auto-enhanced: Strengthened hook opening")
            if sb.scenes and sb.scenes[0].duration_seconds > 3:
                sb.scenes[0].duration_seconds = 2.0
            if not sb.hook_formula:
                sb.hook_formula = "curiosity_gap"

        # Retention below 60%
        if score.retention_arch < 12 and sb:
            score.suggestions.append("Auto-enhanced: Added mid-roll retention hooks")
            total_dur = sum(s.duration_seconds for s in sb.scenes)
            if total_dur > 20:
                mid_idx = len(sb.scenes) // 2
                if mid_idx < len(sb.scenes):
                    sb.scenes[mid_idx].text_overlay = "But wait..."

        # Visual below 60%
        if score.visual_quality < 9 and sb:
            score.suggestions.append("Auto-enhanced: Added visual variety markers")
            for i, scene in enumerate(sb.scenes):
                if not scene.shot_type or scene.shot_type == "static":
                    scene.shot_type = ["medium", "close_up", "wide_establishing"][i % 3]

        # Audio below 60%
        if score.audio_quality < 9 and sb:
            score.suggestions.append("Auto-enhanced: Added music mood")
            if not sb.music_mood:
                sb.music_mood = "lo_fi"

        # Platform below 60%
        if score.platform_opt < 9 and sb:
            spec = get_platform_spec(plan.platform)
            ideal = spec.get("ideal_duration", (30, 60))
            total_dur = sum(s.duration_seconds for s in sb.scenes)
            if total_dur > ideal[1]:
                score.suggestions.append(
                    f"Auto-enhanced: Flagged duration ({total_dur:.0f}s) exceeds "
                    f"ideal ({ideal[1]}s) for {plan.platform}"
                )

        # CTA below 60%
        if score.cta_effectiveness < 9 and sb:
            if not sb.cta_text:
                sb.cta_text = "Follow for more!"
                score.suggestions.append("Auto-enhanced: Added default CTA")

        return plan

    def score_and_enhance(self, plan: VideoPlan, threshold: int = 70) -> tuple:
        """Score plan, auto-enhance if below threshold, re-score."""
        score = self.score(plan)
        if score.total < threshold:
            plan = self.auto_enhance(plan, score)
            score = self.score(plan)
        return plan, score

    # ── Scoring criteria ──────────────────────────────────────────────

    def _score_hook(self, plan: VideoPlan) -> int:
        """Hook strength: /20 — opening power, curiosity, scroll-stop potential."""
        score = 0
        sb = plan.storyboard
        if not sb:
            return 0

        # Has hook formula defined
        if sb.hook_formula and sb.hook_formula in HOOK_FORMULAS:
            formula = HOOK_FORMULAS[sb.hook_formula]
            score += min(formula.get("power", 5), 8)  # Up to 8 from formula power
        else:
            score += 2  # Basic points for having any content

        # First scene timing
        if sb.scenes:
            first = sb.scenes[0]
            if first.duration_seconds <= 2.0:
                score += 4  # Quick hook
            elif first.duration_seconds <= 3.0:
                score += 2
            # Has text overlay (visual hook)
            if first.text_overlay:
                score += 3
            # Has narration
            if first.narration:
                score += 3
        else:
            score += 1

        # Has thumbnail concept
        if sb.thumbnail_concept:
            score += 2

        return min(score, 20)

    def _score_retention(self, plan: VideoPlan) -> int:
        """Retention architecture: /20 — pacing, mid-roll hooks, loop potential."""
        score = 0
        sb = plan.storyboard
        if not sb:
            return 0

        scenes = sb.scenes
        if not scenes:
            return 2

        # Scene count variety
        if len(scenes) >= 4:
            score += 4
        elif len(scenes) >= 2:
            score += 2

        # Scene duration variety (not all same length)
        durations = [s.duration_seconds for s in scenes]
        if len(set(round(d) for d in durations)) >= 2:
            score += 3

        # Shot type variety
        shot_types = [s.shot_type for s in scenes if s.shot_type]
        if len(set(shot_types)) >= 3:
            score += 4
        elif len(set(shot_types)) >= 2:
            score += 2

        # Transition variety
        transitions = [s.transition_in for s in scenes if s.transition_in and s.transition_in != "cut"]
        if transitions:
            score += 3

        # Mid-roll text overlays (retention hooks)
        mid_overlays = [s for s in scenes[1:-1] if s.text_overlay] if len(scenes) > 2 else []
        if mid_overlays:
            score += 3

        # Total duration in ideal range
        total = sum(durations)
        if 25 <= total <= 60:
            score += 3
        elif 15 <= total <= 90:
            score += 1

        return min(score, 20)

    def _score_visual(self, plan: VideoPlan) -> int:
        """Visual quality: /15 — variety, prompts, style consistency."""
        score = 0
        sb = plan.storyboard
        if not sb:
            return 0

        # Has color grade
        if sb.color_grade:
            score += 3

        # Visual prompts in scenes
        scenes_with_visuals = [s for s in sb.scenes if s.visual_prompt]
        if scenes_with_visuals:
            coverage = len(scenes_with_visuals) / max(len(sb.scenes), 1)
            score += round(coverage * 6)

        # Varied shot types
        shots = set(s.shot_type for s in sb.scenes if s.shot_type)
        score += min(len(shots), 3)

        # Thumbnail concept
        if sb.thumbnail_concept:
            score += 3

        return min(score, 15)

    def _score_audio(self, plan: VideoPlan) -> int:
        """Audio quality: /15 — TTS voice, music, SFX, subtitle sync."""
        score = 0
        sb = plan.storyboard
        if not sb:
            return 0

        # Has voice ID
        if sb.voice_id:
            score += 3

        # Has music mood
        if sb.music_mood:
            score += 3

        # Subtitle style defined
        if sb.subtitle_style and sb.subtitle_style in SUBTITLE_STYLES:
            score += 3

        # Narration in scenes
        narrated = [s for s in sb.scenes if s.narration]
        if narrated:
            coverage = len(narrated) / max(len(sb.scenes), 1)
            score += round(coverage * 4)

        # Music cues in scenes
        cues = [s for s in sb.scenes if s.music_cue]
        if cues:
            score += 2

        return min(score, 15)

    def _score_platform(self, plan: VideoPlan) -> int:
        """Platform optimization: /15 — specs compliance, duration, hashtags."""
        score = 0
        sb = plan.storyboard
        if not sb:
            return 0

        spec = get_platform_spec(plan.platform)

        # Duration within limits
        total = sum(s.duration_seconds for s in sb.scenes)
        ideal = spec.get("ideal_duration", (30, 60))
        if ideal[0] <= total <= ideal[1]:
            score += 5
        elif spec.get("min_duration", 0) <= total <= spec.get("max_duration", 600):
            score += 2

        # Has hashtags
        if sb.hashtags:
            max_tags = spec.get("max_hashtags", 15)
            if len(sb.hashtags) <= max_tags:
                score += 3
            else:
                score += 1

        # Platform matches format
        if plan.platform in ("youtube_shorts", "tiktok", "instagram_reels"):
            if plan.format == "short":
                score += 3
        elif plan.platform == "youtube":
            if plan.format == "standard":
                score += 3

        # Title length check
        max_title = spec.get("max_title_chars", 100)
        if len(sb.title) <= max_title:
            score += 2

        # Subtitle presence (critical for short-form)
        if sb.subtitle_style:
            score += 2

        return min(score, 15)

    def _score_cta(self, plan: VideoPlan) -> int:
        """CTA effectiveness: /15 — clear, positioned, niche-appropriate."""
        score = 0
        sb = plan.storyboard
        if not sb:
            return 0

        # Has CTA text
        if sb.cta_text:
            score += 5
            # CTA includes action verb
            action_words = ["follow", "subscribe", "save", "share", "comment",
                            "click", "tap", "watch", "check"]
            if any(w in sb.cta_text.lower() for w in action_words):
                score += 3

        # Last scene has narration (verbal CTA)
        if sb.scenes:
            last = sb.scenes[-1]
            if last.narration:
                score += 3
            if last.text_overlay:
                score += 2

        # Has hashtags (discovery CTA)
        if sb.hashtags:
            score += 2

        return min(score, 15)
