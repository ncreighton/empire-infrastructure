"""SuperPrompt Enhancer — 6-layer video query enhancement.

Layers:
1. Niche knowledge injection (visual DNA, content pillars, brand voice)
2. Platform context (specs, optimal duration, hashtag strategy)
3. Trending/seasonal context (Samhain in Oct, Black Friday in Nov, etc.)
4. Hook formula injection (best formula for this topic)
5. Production depth (retention strategies, engagement triggers)
6. Personalization from VideoCodex (what's worked before)
"""

from datetime import datetime
from ..models import EnhancedQuery
from ..knowledge.niche_profiles import get_niche_profile, get_visual_dna
from ..knowledge.platform_specs import get_platform_spec
from ..knowledge.hook_formulas import get_best_hook, get_hooks_ranked, HOOK_FORMULAS
from ..knowledge.pacing import get_pacing
from ..knowledge.retention_patterns import get_retention_for_platform
from ..knowledge.trending_formats import get_trending_formats
from ..knowledge.music_moods import get_mood_for_niche
from ..knowledge.color_grades import get_color_grade

# Seasonal context by month
_SEASONAL_CONTEXT = {
    1: "New Year fresh starts, winter aesthetics, goal-setting content trending",
    2: "Valentine's Day, Imbolc sabbat, self-love content, winter cozy vibes",
    3: "Spring equinox, Ostara, new beginnings, spring cleaning content",
    4: "Easter, spring refresh, outdoor tech, garden content",
    5: "Beltane, Mother's Day, outdoor season starting, summer prep",
    6: "Summer solstice, Litha, outdoor content, vacation season",
    7: "Mid-year review, summer projects, beach/outdoor content",
    8: "Lughnasadh, back-to-school, harvest themes, fall prep",
    9: "Autumn equinox, Mabon, cozy season starting, fall aesthetics",
    10: "Samhain, Halloween, spooky content at PEAK — witchcraft videos surge 300%",
    11: "Black Friday, gift guides, gratitude content, holiday prep begins",
    12: "Yule, winter solstice, holiday content, year-end reviews, gift guides",
}

# Content type detection keywords
_CONTENT_TYPE_KEYWORDS = {
    "tutorial": ["how to", "tutorial", "guide", "setup", "install", "step by step"],
    "review": ["review", "worth it", "vs", "comparison", "best", "worst"],
    "story": ["story", "legend", "myth", "tale", "history", "once upon"],
    "listicle": ["top", "best", "things", "tips", "hacks", "secrets", "reasons"],
    "news": ["breaking", "new", "released", "update", "announced", "latest"],
    "motivation": ["manifest", "motivation", "inspire", "believe", "mindset"],
    "entertainment": ["funny", "crazy", "insane", "reaction", "challenge"],
}


class PromptEnhancer:
    """Enhances raw video topic queries with 6 layers of context."""

    def __init__(self, codex=None):
        """Optional VideoCodex for personalization layer."""
        self._codex = codex

    def enhance(self, query: str, niche: str,
                platform: str = "youtube_shorts") -> EnhancedQuery:
        """Apply 6 enhancement layers to a raw video topic query."""
        score_before = self._score_query(query)
        layers = []
        enhanced = query.strip()
        niche_context = {}

        # Layer 1: Niche knowledge
        enhanced, ctx = self._inject_niche_knowledge(enhanced, niche)
        layers.append("niche_knowledge")
        niche_context.update(ctx)

        # Layer 2: Platform context
        enhanced = self._inject_platform_context(enhanced, platform)
        layers.append("platform_context")

        # Layer 3: Seasonal/trending
        enhanced = self._inject_seasonal_context(enhanced, niche)
        layers.append("seasonal_trending")

        # Layer 4: Hook formula
        enhanced = self._inject_hook_formula(enhanced, niche)
        layers.append("hook_formula")

        # Layer 5: Production depth
        enhanced = self._inject_production_depth(enhanced, platform)
        layers.append("production_depth")

        # Layer 6: Personalization
        enhanced = self._inject_personalization(enhanced, niche)
        layers.append("personalization")

        score_after = self._score_query(enhanced)

        return EnhancedQuery(
            original=query,
            enhanced=enhanced,
            layers_applied=layers,
            score_before=score_before,
            score_after=score_after,
            niche_context=niche_context,
        )

    def detect_content_type(self, query: str) -> str:
        """Detect the type of content from the query."""
        query_lower = query.lower()
        best_type = "educational"
        best_count = 0

        for ctype, keywords in _CONTENT_TYPE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in query_lower)
            if count > best_count:
                best_count = count
                best_type = ctype

        return best_type

    def _score_query(self, query: str) -> float:
        """Score query richness (0-100)."""
        score = 0.0
        words = query.split()

        # Length
        score += min(len(words) * 2, 20)

        # Has context markers
        context_markers = [
            "[Niche]", "[Platform]", "[Season]", "[Hook]",
            "[Retention]", "[Personalization]",
        ]
        for marker in context_markers:
            if marker in query:
                score += 10

        # Specificity
        if any(c.isdigit() for c in query):
            score += 5
        if len(query) > 100:
            score += 5
        if len(query) > 300:
            score += 5
        if len(query) > 500:
            score += 5

        return min(score, 100)

    # ── Layer implementations ────────────────────────────────────────

    def _inject_niche_knowledge(self, query: str, niche: str) -> tuple:
        """Layer 1: Inject niche visual DNA, content pillars, brand voice."""
        profile = get_niche_profile(niche)
        if not profile:
            return query, {}

        visual_dna = profile.get("visual_dna", {})
        voice = profile.get("voice", {})
        pillars = profile.get("content_pillars", [])

        injection = (
            f"\n\n[Niche: {profile.get('brand', niche)}] "
            f"Aesthetic: {visual_dna.get('aesthetic', 'clean')}. "
            f"Key visuals: {', '.join(visual_dna.get('key_visuals', [])[:5])}. "
            f"Color palette: {', '.join(visual_dna.get('color_palette', [])[:3])}. "
            f"Voice: {voice.get('tone', 'engaging')} — {voice.get('personality', 'expert')}. "
            f"Content pillars: {', '.join(pillars[:4])}. "
            f"AVOID: {', '.join(visual_dna.get('avoid', [])[:3])}."
        )

        ctx = {
            "brand": profile.get("brand", niche),
            "category": profile.get("category", ""),
            "aesthetic": visual_dna.get("aesthetic", ""),
            "tone": voice.get("tone", ""),
        }

        return query + injection, ctx

    def _inject_platform_context(self, query: str, platform: str) -> str:
        """Layer 2: Platform specs, duration, hashtag strategy."""
        spec = get_platform_spec(platform)
        pacing = get_pacing(platform=platform)

        injection = (
            f"\n\n[Platform: {spec['name']}] "
            f"Resolution: {spec['width']}x{spec['height']} ({spec['aspect_ratio']}). "
            f"Duration: {spec.get('ideal_duration', (30, 60))[0]}-{spec.get('ideal_duration', (30, 60))[1]}s ideal. "
            f"Pacing: {pacing.get('cuts_per_minute', 15)} cuts/min, "
            f"{pacing.get('word_rate_wpm', 160)} WPM. "
            f"Hook window: {pacing.get('hook_window_seconds', 1.5)}s. "
            f"Best practices: {'; '.join(spec.get('best_practices', [])[:3])}."
        )

        return query + injection

    def _inject_seasonal_context(self, query: str, niche: str) -> str:
        """Layer 3: Current season, trending angles, relevant events."""
        month = datetime.utcnow().month
        seasonal = _SEASONAL_CONTEXT.get(month, "")
        profile = get_niche_profile(niche)
        category = profile.get("category", "tech")

        trending = get_trending_formats(niche=category)
        trending_names = [t["name"] for t in trending[:3]]

        injection = (
            f"\n\n[Season: Month {month}] {seasonal}. "
            f"Trending formats: {', '.join(trending_names)}."
        )

        return query + injection

    def _inject_hook_formula(self, query: str, niche: str) -> str:
        """Layer 4: Best hook formula with templates."""
        best = get_best_hook(niche)
        formula = HOOK_FORMULAS.get(best, {})
        ranked = get_hooks_ranked(niche)

        templates = formula.get("templates", [])[:2]
        templates_str = " | ".join(templates)

        injection = (
            f"\n\n[Hook: {formula.get('name', best)}] "
            f"Power: {formula.get('power', 7)}/10. "
            f"Templates: {templates_str}. "
            f"Retention anchor: {formula.get('retention_anchor', 'strong opening')}. "
            f"Ranked hooks for this niche: {', '.join(ranked[:4])}."
        )

        return query + injection

    def _inject_production_depth(self, query: str, platform: str) -> str:
        """Layer 5: Retention strategies, engagement patterns."""
        strategies = get_retention_for_platform(platform)
        strat_names = [s["name"] for s in strategies[:3]]
        strat_desc = "; ".join(
            f"{s['name']}: {s['description']}" for s in strategies[:2]
        )

        injection = (
            f"\n\n[Retention] Strategies: {', '.join(strat_names)}. "
            f"Details: {strat_desc}."
        )

        return query + injection

    def _inject_personalization(self, query: str, niche: str) -> str:
        """Layer 6: Past performance data from VideoCodex."""
        if not self._codex:
            return query + "\n\n[Personalization] No history yet — using niche defaults."

        insights = self._codex.get_insights(niche=niche)
        total = insights.get("total_videos", 0)

        if total == 0:
            return query + "\n\n[Personalization] First video for this niche — using niche defaults."

        best_hooks = insights.get("best_hooks", [])
        hook_str = ", ".join(h["hook"] for h in best_hooks[:3]) if best_hooks else "none yet"
        avg_cost = insights.get("avg_cost_per_video", 0)

        injection = (
            f"\n\n[Personalization] {total} videos created for this niche. "
            f"Best hooks: {hook_str}. "
            f"Average cost: ${avg_cost:.3f}/video."
        )

        return query + injection
