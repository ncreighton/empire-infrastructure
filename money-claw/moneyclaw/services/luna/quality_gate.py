"""Response Quality Gate — scores every Luna response before sending.

All rule-based, zero AI cost. 100-point scoring across 4 dimensions:
- Persona adherence (25pts)
- Emotional attunement (25pts)
- Knowledge depth (25pts)
- Actionability (25pts)

If a response scores below the threshold (default 65), it should be
regenerated with improvement hints injected into the prompt.
"""

import re

from .persona import FORBIDDEN_PHRASES, VOICE_TRAITS

# Generic AI phrases that Luna should never use
_GENERIC_AI_PHRASES = [
    "i understand your concern",
    "that's a great question",
    "i appreciate you sharing",
    "it's important to remember",
    "let me help you with that",
    "here are some suggestions",
    "i hope this helps",
    "feel free to ask",
    "don't hesitate to",
    "in conclusion",
    "to summarize",
    "based on the information",
    "as an ai",
    "i'm just a",
    "i cannot predict",
    "for entertainment purposes",
]

# Luna voice markers — things she SHOULD say
_LUNA_VOICE_MARKERS = [
    "darling", "love", "dear one", "sweet one", "beautiful soul",
    "the cards", "the moon", "the stars", "the universe",
    "blessed be", "merry meet", "sacred", "spirit",
    "energy", "ritual", "crystal", "herb",
]

# Empathy markers
_EMPATHY_MARKERS = [
    "i understand", "i see you", "i hear you", "i feel",
    "that must", "how brave", "you're not alone",
    "i'm holding space", "that takes courage",
]

# Action words
_ACTION_MARKERS = [
    "try", "practice", "meditate", "journal", "light a",
    "carry", "place", "breathe", "ground yourself",
    "set an intention", "create", "write down",
    "work with", "cleanse", "charge", "anoint",
]

# Timing markers
_TIMING_MARKERS = [
    "tonight", "tomorrow", "this week", "full moon",
    "new moon", "morning", "evening", "before bed",
    "dawn", "dusk", "midnight", "sunrise", "sunset",
]


class ResponseQualityGate:
    """Scores every Luna response before sending."""

    def __init__(self, threshold: int = 65):
        self.threshold = threshold

    def score(self, response: str, user_message: str = "",
              context: dict | None = None) -> tuple[int, list[dict]]:
        """Score a response. Returns (total_score, list_of_deductions)."""
        deductions = []
        ctx = context or {}

        persona_score = self._score_persona(response, deductions)
        empathy_score = self._score_empathy(response, user_message, ctx, deductions)
        knowledge_score = self._score_knowledge(response, ctx, deductions)
        action_score = self._score_actionability(response, deductions)

        total = persona_score + empathy_score + knowledge_score + action_score
        return total, deductions

    def passes(self, response: str, user_message: str = "",
               context: dict | None = None) -> bool:
        """Check if a response meets the quality threshold."""
        total, _ = self.score(response, user_message, context)
        return total >= self.threshold

    def suggest_improvements(self, response: str, user_message: str = "",
                             context: dict | None = None) -> list[str]:
        """Generate specific improvement suggestions."""
        total, deductions = self.score(response, user_message, context)
        suggestions = []

        for d in deductions:
            if d["dimension"] == "persona" and d["points"] >= 3:
                suggestions.append(f"Remove generic phrase: '{d.get('detail', '')}'")
            elif d["dimension"] == "persona" and "forbidden" in d.get("reason", ""):
                suggestions.append(f"Remove forbidden phrase: '{d.get('detail', '')}'")

        # Category-level suggestions
        categories = {}
        for d in deductions:
            dim = d["dimension"]
            categories[dim] = categories.get(dim, 0) + d["points"]

        if categories.get("persona", 0) >= 8:
            suggestions.append("Use more Luna voice markers (darling, the cards, blessed be, etc.)")
        if categories.get("empathy", 0) >= 8:
            suggestions.append("Add empathy markers (I see you, that must be, you're not alone)")
        if categories.get("knowledge", 0) >= 8:
            suggestions.append("Include specific spiritual references (card names, crystals, herbs, moon phases)")
        if categories.get("actionability", 0) >= 8:
            suggestions.append("Add concrete actions (try this ritual, carry this crystal, journal about)")

        if not suggestions and total < self.threshold:
            suggestions.append("Response feels generic — add personal touches and specific spiritual guidance")

        return suggestions

    def _score_persona(self, response: str, deductions: list) -> int:
        """Score persona adherence (25 points max)."""
        score = 25
        lower = response.lower()

        # Check forbidden phrases (-5 each)
        for phrase in FORBIDDEN_PHRASES:
            if phrase.lower() in lower:
                penalty = min(5, score)
                score -= penalty
                deductions.append({
                    "dimension": "persona",
                    "reason": "forbidden phrase",
                    "detail": phrase,
                    "points": penalty,
                })

        # Check generic AI language (-3 each)
        for phrase in _GENERIC_AI_PHRASES:
            if phrase in lower:
                penalty = min(3, score)
                score -= penalty
                deductions.append({
                    "dimension": "persona",
                    "reason": "generic AI language",
                    "detail": phrase,
                    "points": penalty,
                })

        # Check for Luna voice markers (+3 each, up to score recovery)
        voice_bonus = 0
        for marker in _LUNA_VOICE_MARKERS:
            if marker in lower:
                voice_bonus += 3
        # Voice markers can recover up to 10 points of deductions
        recovery = min(voice_bonus, 25 - score, 10)
        score += recovery

        return max(0, min(25, score))

    def _score_empathy(self, response: str, user_message: str,
                       context: dict, deductions: list) -> int:
        """Score emotional attunement (25 points max)."""
        score = 25
        lower = response.lower()
        user_lower = user_message.lower()

        # Detect if user is expressing distress
        distress_markers = ["sad", "anxious", "scared", "hurt", "angry", "crying",
                           "struggling", "lost", "alone", "broken", "overwhelmed"]
        user_in_distress = any(m in user_lower for m in distress_markers)

        if user_in_distress:
            # Check for empathy markers
            has_empathy = any(m in lower for m in _EMPATHY_MARKERS)
            if not has_empathy:
                deductions.append({
                    "dimension": "empathy",
                    "reason": "missing empathy for distressed user",
                    "detail": "User expressed distress but response lacks empathy markers",
                    "points": 8,
                })
                score -= 8

            # Check for dismissiveness
            dismissive = ["just", "simply", "easy", "get over", "move on", "think positive"]
            for d in dismissive:
                if d in lower:
                    deductions.append({
                        "dimension": "empathy",
                        "reason": "dismissive language",
                        "detail": d,
                        "points": 5,
                    })
                    score -= 5
                    break

        # Response length check relative to depth
        depth = context.get("depth_level", 2)
        word_count = len(response.split())
        min_words = {1: 15, 2: 30, 3: 60, 4: 100}
        if word_count < min_words.get(depth, 30):
            deductions.append({
                "dimension": "empathy",
                "reason": "response too short for depth level",
                "detail": f"{word_count} words at depth {depth} (min {min_words.get(depth, 30)})",
                "points": 5,
            })
            score -= 5

        # Check for question-asking (engagement)
        if depth >= 3 and "?" not in response:
            deductions.append({
                "dimension": "empathy",
                "reason": "no follow-up question at high depth",
                "detail": "Deep conversations should include thoughtful questions",
                "points": 3,
            })
            score -= 3

        return max(0, min(25, score))

    def _score_knowledge(self, response: str, context: dict,
                         deductions: list) -> int:
        """Score knowledge depth (25 points max)."""
        score = 25
        lower = response.lower()

        # Check for specific spiritual references
        has_specific_card = bool(re.search(r"(the\s+)?(tower|fool|lovers|empress|emperor|magician|high\s+priestess|hermit|wheel|justice|hanged|death|temperance|devil|star|moon|sun|judgment|world|ace|two|three|four|five|six|seven|eight|nine|ten|page|knight|queen|king)\b", lower))
        has_crystal = any(c in lower for c in ["amethyst", "quartz", "tourmaline", "citrine", "selenite",
                                                "obsidian", "labradorite", "moonstone", "jasper", "agate"])
        has_herb = any(h in lower for h in ["lavender", "rosemary", "sage", "chamomile", "mugwort",
                                            "basil", "thyme", "cinnamon", "bay", "mint"])
        has_moon = any(m in lower for m in ["full moon", "new moon", "waxing", "waning",
                                            "moon phase", "lunar"])

        specifics = sum([has_specific_card, has_crystal, has_herb, has_moon])
        if specifics == 0:
            deductions.append({
                "dimension": "knowledge",
                "reason": "no specific spiritual references",
                "detail": "Missing card names, crystals, herbs, or moon references",
                "points": 8,
            })
            score -= 8
        elif specifics == 1:
            deductions.append({
                "dimension": "knowledge",
                "reason": "limited spiritual references",
                "detail": "Only one type of reference present",
                "points": 3,
            })
            score -= 3

        # Check for personalization tokens
        user_profile = context.get("user_profile", {})
        if user_profile:
            entities = user_profile.get("entities", {})
            if entities:
                # Check if any entity names appear in response
                all_names = []
                for elist in entities.values():
                    all_names.extend(e.get("entity_name", "") for e in elist)
                has_personal = any(name.lower() in lower for name in all_names if name)
                if not has_personal and len(all_names) > 0:
                    deductions.append({
                        "dimension": "knowledge",
                        "reason": "no personalization",
                        "detail": "Response doesn't reference known user details",
                        "points": 4,
                    })
                    score -= 4

        return max(0, min(25, score))

    def _score_actionability(self, response: str, deductions: list) -> int:
        """Score actionability (25 points max)."""
        score = 25
        lower = response.lower()

        # Check for action markers
        has_actions = sum(1 for a in _ACTION_MARKERS if a in lower)
        if has_actions == 0:
            deductions.append({
                "dimension": "actionability",
                "reason": "no actionable advice",
                "detail": "Response lacks concrete actions or practices",
                "points": 10,
            })
            score -= 10
        elif has_actions == 1:
            deductions.append({
                "dimension": "actionability",
                "reason": "limited actionable advice",
                "detail": "Only one action suggested",
                "points": 4,
            })
            score -= 4

        # Check for timing recommendations
        has_timing = any(t in lower for t in _TIMING_MARKERS)
        if not has_timing:
            deductions.append({
                "dimension": "actionability",
                "reason": "no timing recommendation",
                "detail": "Missing when to take action (tonight, this week, etc.)",
                "points": 3,
            })
            score -= 3

        # Check for follow-up offer
        followup_markers = ["come back", "check in", "let me know", "reach out",
                           "next time", "we can explore", "shall we"]
        has_followup = any(f in lower for f in followup_markers)
        if not has_followup:
            deductions.append({
                "dimension": "actionability",
                "reason": "no follow-up offered",
                "detail": "Missing invitation to continue the conversation",
                "points": 3,
            })
            score -= 3

        return max(0, min(25, score))
