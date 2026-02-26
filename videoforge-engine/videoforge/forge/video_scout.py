"""VideoScout — Topic analysis, niche fit scoring, virality prediction, gap detection."""

from ..models import ScoutResult, ContentPillar
from ..knowledge.niche_profiles import NICHE_PROFILES, get_niche_profile
from ..knowledge.hook_formulas import get_best_hook, get_hooks_ranked, HOOK_FORMULAS
from ..knowledge.trending_formats import get_trending_formats


# Keywords that signal content pillar intent
_PILLAR_KEYWORDS = {
    ContentPillar.EDUCATIONAL: [
        "how to", "what is", "why", "explain", "guide", "learn", "understand",
        "difference between", "basics", "beginner", "introduction", "101",
    ],
    ContentPillar.ENTERTAINMENT: [
        "funny", "crazy", "insane", "you won't believe", "shocking",
        "reaction", "challenge", "prank", "fails",
    ],
    ContentPillar.INSPIRATIONAL: [
        "motivation", "inspire", "transform", "manifest", "believe",
        "power of", "journey", "affirmation", "mindset",
    ],
    ContentPillar.TUTORIAL: [
        "tutorial", "step by step", "how to make", "setup", "install",
        "configure", "build", "create", "diy",
    ],
    ContentPillar.LISTICLE: [
        "top", "best", "worst", "things", "ways", "tips", "hacks",
        "secrets", "mistakes", "reasons",
    ],
    ContentPillar.STORY: [
        "story of", "legend", "myth", "tale", "origin", "history",
        "once upon", "ancient", "saga", "epic",
    ],
    ContentPillar.REVIEW: [
        "review", "worth it", "honest", "should you buy", "vs",
        "comparison", "test", "unboxing", "verdict",
    ],
    ContentPillar.COMPARISON: [
        "vs", "versus", "compared", "difference", "better",
        "which one", "battle", "showdown",
    ],
    ContentPillar.NEWS: [
        "breaking", "just announced", "new", "update", "released",
        "launched", "revealed", "2026", "latest",
    ],
}

# Niche-specific virality booster keywords
_VIRALITY_KEYWORDS = {
    "witchcraft": ["spell", "ritual", "moon", "crystal", "tarot", "hex", "potion",
                    "candle", "herb", "divination", "samhain", "yule", "altar"],
    "mythology": ["god", "goddess", "zeus", "thor", "dragon", "demon", "ancient",
                   "warrior", "underworld", "titan", "olympus", "norse", "greek"],
    "tech": ["smart", "alexa", "google", "automation", "wifi", "bluetooth",
             "setup", "hack", "trick", "upgrade", "deal", "budget"],
    "ai_news": ["chatgpt", "ai", "artificial intelligence", "openai", "anthropic",
                "gpt", "claude", "model", "benchmark", "llm", "agent"],
    "business": ["money", "income", "passive", "side hustle", "free", "tool",
                  "automate", "scale", "profit", "revenue"],
    "lifestyle": ["easy", "quick", "simple", "routine", "hack", "save",
                   "budget", "family", "kids", "home"],
    "fitness": ["workout", "gains", "results", "before after", "transformation",
                "burn", "muscle", "cardio", "tracking"],
    "journal": ["spread", "setup", "tracker", "theme", "washi", "layout",
                "stickers", "monthly", "weekly"],
    "review": ["best", "worst", "honest", "worth", "buy", "avoid",
               "comparison", "budget", "premium"],
}

_NICHE_CATEGORY = {
    "witchcraftforbeginners": "witchcraft",
    "moonrituallibrary": "witchcraft",
    "manifestandalign": "witchcraft",
    "mythicalarchives": "mythology",
    "smarthomewizards": "tech",
    "smarthomegearreviews": "review",
    "pulsegearreviews": "review",
    "wearablegearreviews": "review",
    "aidiscoverydigest": "ai_news",
    "aiinactionhub": "tech",
    "clearainews": "ai_news",
    "wealthfromai": "business",
    "bulletjournals": "journal",
    "theconnectedhaven": "lifestyle",
    "familyflourish": "lifestyle",
    "celebrationseason": "lifestyle",
}


class VideoScout:
    """Analyzes topics for niche fit, virality potential, and content gaps."""

    def analyze(self, topic: str, niche: str, platform: str = "youtube_shorts") -> ScoutResult:
        """Full topic analysis returning ScoutResult."""
        topic_lower = topic.lower()
        profile = get_niche_profile(niche)
        category = _NICHE_CATEGORY.get(niche, "tech")

        # Score niche fit (0-100)
        niche_fit = self._score_niche_fit(topic_lower, niche, profile, category)

        # Score virality (0-100)
        virality = self._score_virality(topic_lower, category, platform)

        # Detect content pillar
        pillar = self._detect_pillar(topic_lower)

        # Suggest hook formula
        hook_key = self._suggest_hook(topic_lower, niche)

        # Visual style from niche profile
        visual_dna = profile.get("visual_dna", {})
        visual_style = visual_dna.get("aesthetic", "clean")

        # Trending format suggestion
        trending = get_trending_formats(niche=category, platform=platform)
        suggested_format = trending[0]["key"] if trending else "faceless_narrator"

        # Extract keywords
        keywords = self._extract_keywords(topic_lower, category)

        # Identify content gaps
        gaps = self._find_gaps(topic_lower, niche, profile)

        # Related topics
        related = self._suggest_related(topic_lower, category)

        # Completeness score
        completeness = self._score_completeness(topic_lower)

        # Warnings
        warnings = self._check_warnings(topic_lower, niche)

        return ScoutResult(
            topic=topic,
            niche=niche,
            niche_fit_score=niche_fit,
            virality_score=virality,
            completeness_score=completeness,
            suggested_hook=hook_key,
            suggested_format=suggested_format,
            suggested_pillar=pillar,
            visual_style=visual_style,
            content_gaps=gaps,
            related_topics=related,
            keywords=keywords,
            warnings=warnings,
        )

    def _score_niche_fit(self, topic: str, niche: str, profile: dict, category: str) -> int:
        """Score how well a topic fits the niche (0-100)."""
        score = 30  # Base score — any topic has some potential

        # Check content pillars
        pillars = profile.get("content_pillars", [])
        for pillar in pillars:
            if pillar.replace("_", " ") in topic:
                score += 15
                break

        # Check niche-specific keywords
        niche_kw = _VIRALITY_KEYWORDS.get(category, [])
        matches = sum(1 for kw in niche_kw if kw in topic)
        score += min(matches * 8, 40)

        # Check visual DNA key visuals
        visuals = profile.get("visual_dna", {}).get("key_visuals", [])
        for v in visuals:
            if v in topic:
                score += 10
                break

        return min(score, 100)

    def _score_virality(self, topic: str, category: str, platform: str) -> int:
        """Score virality potential (0-100)."""
        score = 20  # Base

        # Virality keywords
        kw = _VIRALITY_KEYWORDS.get(category, [])
        matches = sum(1 for k in kw if k in topic)
        score += min(matches * 7, 35)

        # Hook-friendly phrasing
        hook_phrases = ["secret", "nobody", "truth", "wrong", "hack",
                        "mistake", "stop", "never", "always", "best"]
        for hp in hook_phrases:
            if hp in topic:
                score += 5

        # Number in topic (listicle appeal)
        if any(c.isdigit() for c in topic):
            score += 10

        # Question format
        if "?" in topic or topic.startswith(("how", "what", "why", "when")):
            score += 8

        # Short-form platform bonus
        if platform in ("youtube_shorts", "tiktok"):
            score += 5

        return min(score, 100)

    def _detect_pillar(self, topic: str) -> str:
        """Detect the best content pillar for this topic."""
        best_pillar = ContentPillar.EDUCATIONAL.value
        best_count = 0

        for pillar, keywords in _PILLAR_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in topic)
            if count > best_count:
                best_count = count
                best_pillar = pillar.value

        return best_pillar

    def _suggest_hook(self, topic: str, niche: str) -> str:
        """Suggest best hook formula for topic + niche combination."""
        ranked = get_hooks_ranked(niche)

        # Check if topic naturally fits a specific hook
        if any(w in topic for w in ["story", "legend", "myth", "tale", "once"]):
            return "story_hook"
        if any(w in topic for w in ["top", "best", "worst", "things"]):
            return "list_authority"
        if any(w in topic for w in ["wrong", "lie", "myth vs", "actually"]):
            return "contrarian"
        if any(w in topic for w in ["before", "after", "transformation", "tried"]):
            return "before_after"
        if any(c.isdigit() for c in topic):
            return "list_authority"

        return ranked[0] if ranked else "curiosity_gap"

    def _extract_keywords(self, topic: str, category: str) -> list:
        """Extract relevant keywords from topic."""
        niche_kw = _VIRALITY_KEYWORDS.get(category, [])
        found = [kw for kw in niche_kw if kw in topic]
        return found[:10]

    def _find_gaps(self, topic: str, niche: str, profile: dict) -> list:
        """Identify what's missing for a complete video on this topic."""
        gaps = []
        if len(topic.split()) < 3:
            gaps.append("Topic is too vague — add specificity")
        if not any(c.isdigit() for c in topic):
            gaps.append("Consider adding a number for listicle appeal")
        if "?" not in topic and not any(w in topic for w in ["how", "why", "what"]):
            gaps.append("Consider framing as a question for engagement")

        pillars = profile.get("content_pillars", [])
        pillar_match = any(p.replace("_", " ") in topic for p in pillars)
        if not pillar_match:
            gaps.append(f"Topic doesn't directly match niche pillars: {', '.join(pillars[:3])}")

        return gaps

    def _suggest_related(self, topic: str, category: str) -> list:
        """Suggest related topic ideas."""
        kw = _VIRALITY_KEYWORDS.get(category, [])
        words = topic.split()
        related = []

        # Combine topic words with niche keywords
        for k in kw[:5]:
            if k not in topic:
                related.append(f"{' '.join(words[:3])} + {k}")
                if len(related) >= 5:
                    break

        return related

    def _score_completeness(self, topic: str) -> int:
        """Score how complete/specific a topic description is."""
        score = 0
        words = topic.split()
        score += min(len(words) * 5, 30)  # Length
        if any(c.isdigit() for c in topic):
            score += 15  # Has numbers
        if any(w in topic for w in ["for", "about", "using", "with"]):
            score += 15  # Prepositions = specificity
        if len(topic) > 20:
            score += 10
        if len(topic) > 40:
            score += 10
        # Action verbs
        if any(w in topic for w in ["how to", "make", "create", "build", "setup"]):
            score += 20
        return min(score, 100)

    def _check_warnings(self, topic: str, niche: str) -> list:
        """Check for potential issues with this topic."""
        warnings = []
        profile = get_niche_profile(niche)
        avoid = profile.get("voice", {}).get("avoid", [])
        for a in avoid:
            if a.lower() in topic:
                warnings.append(f"Topic contains avoided term for this niche: '{a}'")

        visual_avoid = profile.get("visual_dna", {}).get("avoid", [])
        for va in visual_avoid:
            if va.lower() in topic:
                warnings.append(f"Topic may trigger avoided visual style: '{va}'")

        if len(topic) > 100:
            warnings.append("Topic is very long — consider shortening for clarity")

        return warnings
