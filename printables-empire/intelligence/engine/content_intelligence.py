"""Master intelligence orchestrator for content generation.

Routes requests through topic_scout, seo_optimizer, and content_scorer.
Injects voice, SEO, and seasonal context into every content piece.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from content.models import ContentType
from intelligence.engine.topic_scout import TopicScout
from intelligence.engine.seo_optimizer import SEOOptimizer
from intelligence.engine.content_scorer import ContentScorer
from intelligence.engine.content_calendar import ContentCalendar


class RequestType(Enum):
    ARTICLE = "article"
    REVIEW = "review"
    LISTING = "listing"
    POST = "post"


@dataclass
class EnhancementContext:
    """Context gathered by intelligence modules before content generation."""
    request_type: RequestType
    raw_input: str
    topic: str = ""
    keywords: list[str] = field(default_factory=list)
    difficulty: str = "beginner"
    seo_data: dict = field(default_factory=dict)
    seasonal_context: str = ""
    voice_profile: str = "maker_mentor"
    enhancements_applied: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ContentIntelligence:
    """Master orchestrator — enhances content requests with intelligence."""

    def __init__(self):
        self.scout = TopicScout()
        self.seo = SEOOptimizer()
        self.scorer = ContentScorer()
        self.calendar = ContentCalendar()

    def enhance(self, raw_input: str, content_type: str, **kwargs) -> EnhancementContext:
        """Enhance a content request with intelligence context.

        Args:
            raw_input: The topic or product name
            content_type: article, review, listing, or post
            **kwargs: Additional context (keywords, difficulty, etc.)
        """
        req_type = RequestType(content_type)
        ctx = EnhancementContext(
            request_type=req_type,
            raw_input=raw_input,
            topic=raw_input,
        )

        # 1. Topic research
        topic_data = self.scout.research(raw_input, content_type)
        ctx.keywords = topic_data.get("keywords", kwargs.get("keywords", []))
        ctx.difficulty = topic_data.get("difficulty", kwargs.get("difficulty", "beginner"))
        ctx.enhancements_applied.append("topic_research")

        # 2. SEO optimization
        seo_data = self.seo.optimize(raw_input, content_type, ctx.keywords)
        ctx.seo_data = seo_data
        ctx.enhancements_applied.append("seo_optimization")

        # 3. Seasonal context
        seasonal = self.calendar.get_seasonal_context()
        ctx.seasonal_context = seasonal
        ctx.enhancements_applied.append("seasonal_context")

        # 4. Voice profile selection
        voice_map = {
            RequestType.ARTICLE: "maker_mentor",
            RequestType.REVIEW: "gear_reviewer",
            RequestType.LISTING: "maker_mentor",
            RequestType.POST: "community_voice",
        }
        ctx.voice_profile = kwargs.get("voice_profile", voice_map.get(req_type, "maker_mentor"))

        return ctx

    def score_content(self, content_text: str, content_type: str, keywords: list[str]) -> dict:
        """Score a piece of content. Returns score dict with overall, breakdown, and verdict."""
        return self.scorer.score(content_text, content_type, keywords)

    def get_weekly_plan(self) -> list[dict]:
        """Get the content calendar for the current week."""
        return self.calendar.weekly_plan()

    def suggest_topics(self, content_type: str, count: int = 10) -> list[dict]:
        """Suggest topics for a given content type."""
        return self.scout.suggest_topics(content_type, count)
