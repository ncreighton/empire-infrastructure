"""Witchcraft Video Pipeline — Auto-generates YouTube Shorts from Grimoire + VideoForge.

Combines:
- Grimoire Intelligence (port 8080) for witchcraft knowledge + content ideas
- VideoForge Engine (port 8090) for video creation + rendering
- Brain Intelligence for topic selection and performance tracking

Usage:
    from connectors.witchcraft_video_pipeline import WitchcraftVideoPipeline

    pipeline = WitchcraftVideoPipeline()

    # Generate topic ideas based on current moon phase
    topics = pipeline.generate_topics(count=5)

    # Create a video from a topic (dry run — no render)
    result = pipeline.create_video("Full Moon Protection Ritual", dry_run=True)

    # Full pipeline: topic → script → visuals → render → publish
    result = pipeline.create_video("Full Moon Protection Ritual", render=True, publish=True)

    # Batch: generate topics + create videos
    results = pipeline.batch_create(count=3, render=False)

    # Get content calendar combining grimoire sabbats + moon phases
    calendar = pipeline.get_content_calendar()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GRIMOIRE_BASE = "http://localhost:8080"
VIDEOFORGE_BASE = "http://localhost:8090"
NICHE = "witchcraft"
PLATFORM = "youtube_shorts"


class WitchcraftVideoPipeline:
    """Automated YouTube Shorts pipeline for the Witchcraft channel.

    Generates topic ideas from Grimoire's knowledge base (herbs, crystals,
    moon phases, spells, rituals, tarot) and feeds them to VideoForge for
    video creation with the Witchcraft niche profile (Drew voice, mystical
    color grading, magical transitions).
    """

    # Topic templates that combine grimoire knowledge areas
    TOPIC_TEMPLATES = [
        # Moon-based (timely, recurring)
        "{moon_phase} Ritual: What to Do Tonight",
        "3 {moon_phase} Spells You Need to Try",
        "{moon_phase} Energy: Best Practices for {best_for}",

        # Herb/Crystal spotlights
        "The Secret Power of {herb} in Magick",
        "{crystal}: The Crystal Every Witch Needs",
        "3 Herbs for {intention} Spells",

        # Spell how-tos
        "Quick {intention} Spell for Beginners",
        "How to Cast a {spell_type} Spell (Step by Step)",
        "{intention} Jar Spell That Actually Works",

        # Sabbat content (seasonal)
        "{sabbat} Altar Setup Guide",
        "5 Things to Do for {sabbat}",
        "{sabbat} Ritual: Connect With the Season",

        # Tarot
        "What Your Tarot Card Means: {tarot_card}",
        "3-Card Tarot Spread for {intention}",

        # Educational
        "Witch Tip: {tip}",
        "Beginner Witch Mistakes to Avoid",
        "How Moon Phases Affect Your Magick",
    ]

    INTENTIONS = [
        "protection", "love", "prosperity", "healing",
        "banishing", "divination", "cleansing", "confidence",
    ]

    SPELL_TYPES = [
        "candle", "jar", "sigil", "bath", "mirror",
        "knot", "herb bundle", "crystal grid",
    ]

    WITCH_TIPS = [
        "Always cast a circle before spellwork",
        "Charge your crystals under the full moon",
        "Keep a moon journal to track your cycles",
        "Ground yourself before and after rituals",
        "Your intention is the most powerful ingredient",
        "Salt is the ultimate cleansing tool",
        "Bay leaves are perfect for wish magick",
        "Cinnamon attracts prosperity and success",
    ]

    def __init__(
        self,
        grimoire_url: str = GRIMOIRE_BASE,
        videoforge_url: str = VIDEOFORGE_BASE,
    ):
        self.grimoire_url = grimoire_url.rstrip("/")
        self.videoforge_url = videoforge_url.rstrip("/")

    def _grimoire(self, method: str, endpoint: str, **kwargs) -> dict | None:
        try:
            resp = requests.request(method, f"{self.grimoire_url}{endpoint}",
                                    timeout=10, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Grimoire API: %s %s -> %s", method, endpoint, e)
            return None

    def _videoforge(self, method: str, endpoint: str, **kwargs) -> dict | None:
        try:
            resp = requests.request(method, f"{self.videoforge_url}{endpoint}",
                                    timeout=120, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("VideoForge API: %s %s -> %s", method, endpoint, e)
            return None

    # ── Topic Generation ──────────────────────────────────────────────

    def generate_topics(self, count: int = 5) -> list[dict]:
        """Generate video topic ideas using current Grimoire energy + knowledge.

        Combines:
        - Current moon phase and energy (timely content)
        - Grimoire herb/crystal knowledge (evergreen content)
        - Seasonal sabbat awareness (seasonal content)
        - VideoForge trending formats
        """
        topics = []

        # Get current energy from Grimoire
        energy = self._grimoire("GET", "/energy")

        # Get upcoming forecast
        forecast = self._grimoire("GET", "/forecast")

        # Build topic context
        moon_phase = energy.get("moon_phase", "Waning Moon") if energy else "Full Moon"
        best_for_list = energy.get("best_for", ["protection"]) if energy else ["protection"]
        zodiac = energy.get("zodiac_sign", "unknown") if energy else "unknown"

        # Get sabbat info (can be string like "Ostara in 14 days" or dict)
        sabbat_raw = forecast.get("upcoming_sabbat", "Ostara") if forecast else "Ostara"
        if isinstance(sabbat_raw, dict):
            current_sabbat = sabbat_raw.get("name", "Ostara")
        elif isinstance(sabbat_raw, str):
            current_sabbat = sabbat_raw.split(" in ")[0].strip()
        else:
            current_sabbat = "Ostara"

        import random
        random.seed()

        # Moon-phase topics (most timely)
        for best_for in best_for_list[:2]:
            topics.append({
                "topic": f"{moon_phase} Energy: {best_for.title()}",
                "category": "moon_content",
                "timeliness": "high",
                "moon_phase": moon_phase,
            })

        # Intention-based topics
        for intention in random.sample(self.INTENTIONS, min(3, count)):
            spell_type = random.choice(self.SPELL_TYPES)
            topics.append({
                "topic": f"Quick {intention.title()} {spell_type.title()} Spell",
                "category": "spell_tutorial",
                "timeliness": "evergreen",
                "intention": intention,
            })

        # Seasonal/sabbat topics
        topics.append({
            "topic": f"{current_sabbat} Ritual: Connect With the Season",
            "category": "sabbat_content",
            "timeliness": "seasonal",
            "sabbat": current_sabbat,
        })

        # Witch tips
        tip = random.choice(self.WITCH_TIPS)
        topics.append({
            "topic": f"Witch Tip: {tip}",
            "category": "educational",
            "timeliness": "evergreen",
        })

        # Score and sort by timeliness
        priority_map = {"high": 3, "seasonal": 2, "evergreen": 1}
        for t in topics:
            t["priority"] = priority_map.get(t["timeliness"], 1)

        topics.sort(key=lambda x: x["priority"], reverse=True)
        return topics[:count]

    # ── Video Creation ────────────────────────────────────────────────

    def create_video(
        self,
        topic: str,
        *,
        dry_run: bool = True,
        render: bool = False,
        publish: bool = False,
    ) -> dict:
        """Create a witchcraft YouTube Short from a topic.

        Steps:
        1. Consult Grimoire for content context
        2. Send to VideoForge with witchcraft niche profile
        3. Optionally render via Creatomate
        4. Optionally publish to YouTube/TikTok

        Args:
            topic: Video topic (e.g., "Full Moon Protection Ritual")
            dry_run: If True, analyze only (no render). Overrides render/publish.
            render: If True, render the video via Creatomate
            publish: If True, publish to configured channels
        """
        # Step 1: Get grimoire context for the topic
        grimoire_context = self._grimoire("POST", "/consult", json={"query": topic})
        energy = self._grimoire("GET", "/energy")

        # Step 2: Enrich topic with grimoire knowledge
        enriched_topic = topic
        if grimoire_context and grimoire_context.get("timing_advice"):
            enriched_topic += f". {grimoire_context['timing_advice']}"

        # Step 3: Send to VideoForge
        if dry_run:
            result = self._videoforge("POST", "/analyze", json={
                "topic": enriched_topic,
                "niche": NICHE,
                "platform": PLATFORM,
            })
        else:
            result = self._videoforge("POST", "/create", json={
                "topic": enriched_topic,
                "niche": NICHE,
                "platform": PLATFORM,
                "format": "short",
                "render": render,
                "publish": publish,
            })

        if not result:
            return {
                "status": "error",
                "topic": topic,
                "error": "VideoForge API unavailable",
            }

        # Step 4: Log to brain
        self._log_to_brain(topic, result, dry_run)

        return {
            "status": "analyzed" if dry_run else "created",
            "topic": topic,
            "enriched_topic": enriched_topic,
            "grimoire_context": {
                "moon_phase": energy.get("moon_phase") if energy else None,
                "best_for": energy.get("best_for", [])[:3] if energy else [],
                "correspondences": grimoire_context.get("correspondences", {}) if grimoire_context else {},
            },
            "videoforge_result": result,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def batch_create(
        self,
        count: int = 3,
        *,
        render: bool = False,
        publish: bool = False,
    ) -> dict:
        """Generate topics and create videos in batch.

        Args:
            count: Number of videos to create
            render: Whether to render videos
            publish: Whether to publish videos
        """
        topics = self.generate_topics(count)
        results = []

        for topic_info in topics:
            result = self.create_video(
                topic_info["topic"],
                dry_run=not render,
                render=render,
                publish=publish,
            )
            result["topic_info"] = topic_info
            results.append(result)

        return {
            "batch_size": len(results),
            "rendered": render,
            "published": publish,
            "results": results,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Content Calendar ──────────────────────────────────────────────

    def get_content_calendar(self, days: int = 14) -> dict:
        """Get a content calendar combining grimoire sabbats + VideoForge calendar.

        Merges:
        - Grimoire forecast (upcoming sabbats, moon phases, energy)
        - VideoForge content calendar (optimal posting times, trending formats)
        """
        # Grimoire forecast
        forecast = self._grimoire("GET", "/forecast")

        # VideoForge calendar
        vf_calendar = self._videoforge("GET", f"/calendar/{NICHE}")

        # Generate topic suggestions for each upcoming key date
        topic_suggestions = []
        if forecast:
            # Moon phase events
            upcoming_phases = forecast.get("upcoming_phases", [])
            for phase in upcoming_phases[:4]:
                phase_name = phase.get("phase", "Full Moon")
                topic_suggestions.append({
                    "date": phase.get("date", "upcoming"),
                    "type": "moon_phase",
                    "topic": f"{phase_name}: What Magick to Work Tonight",
                    "priority": "high" if "Full" in phase_name or "New" in phase_name else "medium",
                })

            # Sabbat events
            sabbat_raw = forecast.get("upcoming_sabbat", "")
            if sabbat_raw:
                sabbat_name = sabbat_raw.split(" in ")[0] if isinstance(sabbat_raw, str) else sabbat_raw.get("name", "Sabbat")
                topic_suggestions.append({
                    "date": "upcoming",
                    "type": "sabbat",
                    "topic": f"{sabbat_name} Celebration Guide",
                    "priority": "high",
                })

        return {
            "grimoire_forecast": forecast,
            "videoforge_calendar": vf_calendar,
            "topic_suggestions": topic_suggestions,
            "days_covered": days,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Cost Estimation ───────────────────────────────────────────────

    def estimate_costs(self, count: int = 1) -> dict:
        """Estimate costs for producing witchcraft videos."""
        estimate = self._videoforge("POST", "/cost-estimate", json={
            "topic": "Sample witchcraft topic",
            "niche": NICHE,
            "platform": PLATFORM,
            "format": "short",
        })

        per_video = estimate.get("estimated_cost", 0.5) if estimate else 0.5

        return {
            "per_video": per_video,
            "batch_cost": round(per_video * count, 2),
            "count": count,
            "includes": ["AI visuals", "ElevenLabs voice (Drew)", "Creatomate render"],
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _log_to_brain(self, topic: str, result: dict, dry_run: bool):
        """Log video creation event to brain database."""
        try:
            from knowledge.brain_db import get_db
            conn = get_db()
            conn.execute("""
                INSERT INTO events (event_type, data, source, timestamp)
                VALUES (?, ?, 'witchcraft_video_pipeline', ?)
            """, (
                "video.analyzed" if dry_run else "video.created",
                json.dumps({"topic": topic, "niche": NICHE})[:500],
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("Brain logging skipped: %s", e)
