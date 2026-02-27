"""VideoForgeEngine — Master orchestrator wiring all intelligence modules together."""

import logging
from datetime import datetime
from pathlib import Path

from .models import (
    VideoPlan, VideoForgeResult, ScoutResult, SentinelScore,
    AmplifyResult, EnhancedQuery, CostBreakdown,
)
from .forge.video_scout import VideoScout
from .forge.video_sentinel import VideoSentinel
from .forge.video_oracle import VideoOracle
from .forge.video_smith import VideoSmith
from .forge.video_codex import VideoCodex
from .amplify.amplify_pipeline import AmplifyPipeline
from .enhancer.prompt_enhancer import PromptEnhancer
from .assembly.script_engine import ScriptEngine
from .assembly.visual_engine import VisualEngine
from .assembly.audio_engine import AudioEngine
from .assembly.subtitle_engine import SubtitleEngine
from .assembly.render_engine import RenderEngine
from .assembly.publisher import Publisher

logger = logging.getLogger(__name__)


class VideoForgeEngine:
    """Master orchestrator — the single entry point for video creation."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "codex.db")

        # FORGE modules
        self.scout = VideoScout()
        self.sentinel = VideoSentinel()
        self.oracle = VideoOracle()
        self.smith = VideoSmith(db_path=db_path)
        self.codex = VideoCodex(db_path=db_path)

        # AMPLIFY
        self.amplify = AmplifyPipeline(db_path=db_path)

        # Enhancer
        self.enhancer = PromptEnhancer(codex=self.codex)

        # Assembly
        self.script_engine = ScriptEngine()
        self.visual_engine = VisualEngine()
        self.audio_engine = AudioEngine()
        self.subtitle_engine = SubtitleEngine()
        self.render_engine = RenderEngine()
        self.publisher = Publisher()

    # ── High-level operations ─────────────────────────────────────────

    def analyze_topic(self, topic: str, niche: str,
                      platform: str = "youtube_shorts") -> VideoForgeResult:
        """Analyze a topic without creating anything. Returns analysis only."""
        # Enhance the query
        enhanced = self.enhancer.enhance(topic, niche, platform)

        # Scout analysis
        scout_result = self.scout.analyze(topic, niche, platform)

        # Oracle recommendation
        oracle_rec = self.oracle.recommend(niche, platform)

        return VideoForgeResult(
            action="analyze",
            scout_result=scout_result,
            enhanced_query=enhanced,
            status="success",
        )

    def create_video(self, topic: str, niche: str,
                     platform: str = "youtube_shorts",
                     format: str = "short",
                     render: bool = True,
                     publish: bool = False) -> VideoForgeResult:
        """Full video creation pipeline.

        Steps:
        [1] Enhance query
        [2] Scout analysis
        [3] Craft storyboard (VideoSmith)
        [4] AMPLIFY pipeline
        [5] Score + enhance (Sentinel)
        [6] Generate AI script (ScriptEngine)
        [7] Generate visual assets (VisualEngine) — FAL.ai per scene
        [8] Generate narration audio (AudioEngine) — ElevenLabs per scene
        [9] Generate subtitles (SubtitleEngine)
        [10] Build RenderScript with real assets, audio, animations
        [11] Submit to Creatomate → render URL
        [12] Log to VideoCodex
        """
        errors = []

        # [1] Enhance the query
        logger.info(f"[1/12] Enhancing query: {topic}")
        enhanced = self.enhancer.enhance(topic, niche, platform)

        # [2] Scout analysis
        logger.info("[2/12] Scouting topic...")
        scout_result = self.scout.analyze(topic, niche, platform)

        # [3] Generate storyboard via VideoSmith
        logger.info("[3/12] Crafting storyboard...")
        plan = self.smith.to_video_plan(topic, niche, platform, format)

        # [4] AMPLIFY the plan
        logger.info("[4/12] Amplifying plan...")
        amplify_result = self.amplify.amplify(plan)
        plan = amplify_result.plan

        # [5] Score with Sentinel
        logger.info("[5/12] Scoring quality...")
        plan, sentinel_score = self.sentinel.score_and_enhance(plan, threshold=70)

        # [6] Generate AI script (enhances storyboard narration)
        logger.info("[6/12] Generating script...")
        script = self.script_engine.generate_script(plan.storyboard)
        plan.script = script

        # Write AI script back to scene narration (replaces template filler)
        if script.word_count > 0 and script.model_used != "fallback_storyboard":
            all_segments = [script.hook] + script.body_segments + [script.cta]
            for i, scene in enumerate(plan.storyboard.scenes):
                if i < len(all_segments) and all_segments[i]:
                    scene.narration = all_segments[i]
                    scene.subtitle_text = all_segments[i]
            # Recalculate scene durations with new narration
            from .knowledge.pacing import get_pacing
            pacing = get_pacing(platform=platform, niche=niche)
            self.smith._calculate_scene_durations(plan.storyboard.scenes, pacing)
            plan.storyboard.total_duration = sum(
                s.duration_seconds for s in plan.storyboard.scenes
            )
            plan.status = "scripted"
            logger.info(f"AI script applied: {script.model_used}, {script.word_count} words")

        # [6.5] Update visual prompts from AI visual directions
        if script.visual_directions:
            updated_count = 0
            for i, scene in enumerate(plan.storyboard.scenes):
                if i < len(script.visual_directions) and script.visual_directions[i]:
                    scene.visual_prompt = script.visual_directions[i]
                    updated_count += 1
            if updated_count > 0:
                logger.info(f"Visual prompts updated from AI directions: {updated_count} scenes")

        # [7] Generate visual assets (FAL.ai for each scene)
        logger.info("[7/12] Generating visual assets...")
        routing = plan.optimizations.get("asset_routing", [])
        visual_assets = self.visual_engine.generate_assets(plan.storyboard, routing)
        plan.visual_assets = visual_assets

        # [8] Generate narration audio (ElevenLabs per scene)
        logger.info("[8/12] Generating narration audio...")
        narration_data = self.audio_engine.generate_full_narration(
            plan.storyboard, niche
        )
        plan.narration_audio_data = narration_data

        # [9] Generate subtitles
        logger.info("[9/12] Generating subtitles...")
        subtitle_track = self.subtitle_engine.generate(plan.storyboard)
        plan.subtitle_track = subtitle_track

        # [10-11] Render (if requested)
        render_result = None
        if render:
            logger.info("[10/12] Building RenderScript...")
            logger.info("[11/12] Rendering video...")
            render_result = self.render_engine.render(plan, wait=True)
            plan.render_id = render_result.get("render_id", "")
            plan.render_url = render_result.get("url", "")
            plan.status = "rendered" if render_result.get("url") else "render_failed"
        else:
            logger.info("[10-11/12] Skipping render (dry run)")
            plan.status = "assembled"

        # [12] Calculate cost + log
        logger.info("[12/12] Logging to codex...")
        cost = self.render_engine.estimate_cost(plan)
        if script.cost > 0:
            cost.script_cost = script.cost

        # Add actual ElevenLabs cost
        audio_cost = 0.0
        for aud in narration_data:
            audio_cost += self.audio_engine.estimate_elevenlabs_cost(aud.get("text", ""))
        cost.audio_cost = round(audio_cost, 4)

        cost.total_cost = round(
            cost.script_cost + cost.visual_cost + cost.audio_cost + cost.render_cost, 4
        )
        plan.cost = cost

        # Log to codex
        video_id = self.codex.log_video(
            topic=topic,
            niche=niche,
            platform=platform,
            format=format,
            hook_formula=plan.storyboard.hook_formula if plan.storyboard else "",
            quality_score=sentinel_score.total,
            total_cost=cost.total_cost,
            render_url=plan.render_url,
            status=plan.status,
        )

        # Log costs
        if cost.script_cost > 0:
            self.codex.log_cost("script", cost.script_cost, provider=script.model_used, video_id=video_id)
        if cost.visual_cost > 0:
            self.codex.log_cost("visual", cost.visual_cost, video_id=video_id)
        if cost.audio_cost > 0:
            self.codex.log_cost("audio", cost.audio_cost, provider="elevenlabs", video_id=video_id)
        if cost.render_cost > 0:
            self.codex.log_cost("render", cost.render_cost, provider="creatomate", video_id=video_id)

        # Publish (if requested)
        if publish and plan.render_url:
            logger.info("Publishing video...")
            pub_result = self.publisher.publish(plan)
            plan.status = "published"

        return VideoForgeResult(
            action="create",
            plan=plan,
            scout_result=scout_result,
            sentinel_score=sentinel_score,
            amplify_result=amplify_result,
            enhanced_query=enhanced,
            cost=cost,
            render_url=plan.render_url,
            status="success",
            errors=errors,
        )

    def generate_topics(self, niche: str, count: int = 10,
                        content_type: str = "educational") -> list:
        """Generate topic ideas for a niche."""
        return self.script_engine.generate_topics(niche, count, content_type)

    def get_calendar(self, niche: str, platform: str = "youtube_shorts") -> dict:
        """Get a 7-day content calendar."""
        rec = self.oracle.recommend(niche, platform)
        return {
            "niche": niche,
            "platform": platform,
            "best_post_time": rec.best_post_time,
            "frequency": rec.frequency_recommendation,
            "calendar": rec.content_calendar,
            "seasonal_angle": rec.seasonal_angle,
            "trending_formats": rec.trending_formats,
        }

    def get_insights(self, niche: str = None) -> dict:
        """Get performance and cost insights."""
        return self.codex.get_insights(niche=niche)

    def estimate_cost(self, topic: str, niche: str,
                      platform: str = "youtube_shorts",
                      format: str = "short") -> dict:
        """Estimate cost before creating a video."""
        plan = self.smith.to_video_plan(topic, niche, platform, format)
        self.amplify._optimize(plan)
        cost = self.render_engine.estimate_cost(plan)
        return {
            "script_cost": cost.script_cost,
            "visual_cost": cost.visual_cost,
            "audio_cost": cost.audio_cost,
            "render_cost": cost.render_cost,
            "total_cost": cost.total_cost,
            "asset_count": cost.asset_count,
        }
