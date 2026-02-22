"""
Mystic Prompt Enhancer — auto-enhances every spiritual query.

Adapts the SuperPromptEngine pattern for witchcraft queries:
  - Auto-detects query type (spell, divination, herb, sabbat, etc.)
  - Injects traditional knowledge context from the correspondences DB
  - Adds current moon phase and seasonal/sabbat context
  - Layers historical and depth/nuance guidance
  - Scores queries before and after enhancement

Every spiritual query that enters this engine exits richer and deeper.
"""

import datetime
from grimoire.models import EnhancedQuery, QueryType
from grimoire.knowledge.correspondences import (
    get_correspondences_for_intention,
    HERBS,
    CRYSTALS,
)
from grimoire.knowledge.moon_phases import (
    calculate_moon_phase,
    get_phase_data,
    MOON_PHASES,
)
from grimoire.knowledge.wheel_of_year import (
    get_seasonal_context,
    get_next_sabbat,
)
from grimoire.knowledge.spell_templates import SPELL_TYPES
from grimoire.knowledge.tarot import get_cards_for_intention
from grimoire.voice import VOICE_PROFILE, apply_voice


# ══════════════════════════════════════════════════════════════════════════════
# QUERY PATTERN MAPS
# ══════════════════════════════════════════════════════════════════════════════

QUERY_PATTERNS: dict[QueryType, list[str]] = {
    QueryType.SPELL_REQUEST: [
        "spell", "cast", "ritual for", "working for", "magick for",
        "how do i enchant", "charm for",
    ],
    QueryType.DIVINATION_QUESTION: [
        "future", "tarot", "oracle", "reading", "divination", "scry",
        "what will", "should i",
    ],
    QueryType.HERB_CRYSTAL_QUERY: [
        "herb", "crystal", "stone", "plant", "flower", "essential oil",
        "which crystal", "what herb",
    ],
    QueryType.SABBAT_PLANNING: [
        "sabbat", "samhain", "yule", "imbolc", "ostara", "beltane",
        "litha", "lughnasadh", "mabon", "wheel of year", "solstice",
        "equinox",
    ],
    QueryType.SHADOW_WORK: [
        "shadow", "inner work", "trauma", "healing past", "inner child",
        "unconscious", "dark night",
    ],
    QueryType.MEDITATION_REQUEST: [
        "meditat", "visualiz", "journey", "guided", "mindful",
        "breathwork", "trance",
    ],
    QueryType.MOON_QUERY: [
        "moon", "lunar", "full moon", "new moon", "waxing", "waning",
        "esbat",
    ],
    QueryType.TAROT_QUERY: [
        "tarot", "card", "spread", "reading", "major arcana",
        "minor arcana", "deck",
    ],
    # GENERAL_WITCHCRAFT is the fallback — no keywords needed
}


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE INJECTION TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_INJECTIONS: dict[QueryType, str] = {
    QueryType.SPELL_REQUEST: (
        "Traditional correspondences for {intention}: "
        "Herbs: {herbs}. Crystals: {crystals}. Colors: {colors}. "
        "Best timing: {timing}. {safety_notes}"
    ),
    QueryType.HERB_CRYSTAL_QUERY: (
        "Magical properties of {subject}: {properties}. "
        "Element: {element}. Planet: {planet}. "
        "Pairs well with: {pairs}. Safety: {safety}"
    ),
    QueryType.SABBAT_PLANNING: (
        "Key {sabbat} correspondences: Colors: {colors}. "
        "Herbs: {herbs}. Themes: {themes}. "
        "Traditional activities: {activities}"
    ),
    QueryType.MOON_QUERY: (
        "Current moon: {phase} in {sign}. Energy: {energy}. "
        "Best for: {best_for}. Avoid: {avoid}."
    ),
    QueryType.DIVINATION_QUESTION: (
        "Divinatory context: {method} tradition suggests focusing on {focus}. "
        "Related cards/symbols: {symbols}. "
        "Best timing for divination: {timing}."
    ),
    QueryType.SHADOW_WORK: (
        "Shadow work context: This work relates to {theme}. "
        "Supportive herbs: {herbs}. Grounding crystals: {crystals}. "
        "Recommended moon phase: waning or dark moon."
    ),
    QueryType.MEDITATION_REQUEST: (
        "Meditation support: Element focus: {element}. "
        "Supportive crystals: {crystals}. Incense/herbs: {herbs}. "
        "Best time of day: {time_of_day}."
    ),
    QueryType.TAROT_QUERY: (
        "Tarot context: Related cards for this intention: {cards}. "
        "Recommended spread: {spread}. "
        "Enhancing correspondences: crystals {crystals}, herbs {herbs}."
    ),
    QueryType.GENERAL_WITCHCRAFT: (
        "Relevant correspondences: Herbs: {herbs}. Crystals: {crystals}. "
        "Colors: {colors}. Timing: {timing}."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL NOTES
# ══════════════════════════════════════════════════════════════════════════════

HISTORICAL_NOTES: dict[str, str] = {
    "candle_magick": (
        "Candle magick has roots in sympathetic magic traditions dating "
        "back millennia, from ancient Roman temple offerings to medieval "
        "European folk practices. The colour, flame behaviour, and wax "
        "remnants were all read as omens."
    ),
    "herb_craft": (
        "Herbal magick is one of the oldest forms of folk practice, "
        "documented in the Ebers Papyrus of ancient Egypt (c. 1550 BCE), "
        "the Greek herbal of Dioscorides, and countless medieval "
        "herbals across Europe and the British Isles."
    ),
    "moon_magick": (
        "Lunar worship and moon-timed practices appear across virtually "
        "every culture, from Mesopotamian moon-god temples to the Celtic "
        "esbat tradition. The synodic month has governed planting, "
        "harvesting, and ritual calendars for millennia."
    ),
    "divination": (
        "Divination practices span thousands of years, from the Oracle "
        "at Delphi to Norse rune casting, Chinese I Ching, and the "
        "tarot tradition that emerged in 15th-century Italy before "
        "becoming a divinatory tool in 18th-century France."
    ),
    "sabbats": (
        "The Wheel of the Year combines Celtic fire festivals (Samhain, "
        "Imbolc, Beltane, Lughnasadh) with Germanic solstice and equinox "
        "celebrations (Yule, Ostara, Litha, Mabon), codified into the "
        "modern eightfold calendar by Gerald Gardner and Ross Nichols "
        "in the mid-20th century."
    ),
    "crystal_work": (
        "Crystal healing traditions appear in ancient Egyptian, Greek, "
        "Chinese, and Indigenous cultures. The Egyptians buried their dead "
        "with lapis lazuli and carnelian; the Greeks named amethyst as a "
        "ward against intoxication. Modern crystal practice draws from "
        "these deep roots."
    ),
    "shadow_work": (
        "Shadow work derives from Carl Jung's concept of the shadow self "
        "— the hidden, repressed aspects of the psyche. In magical "
        "traditions, it parallels the descent of Inanna, Persephone's "
        "journey to the underworld, and the initiatory death-rebirth "
        "cycle found in mystery schools worldwide."
    ),
    "meditation": (
        "Guided visualization and trance work in witchcraft traditions "
        "draws from shamanic journeying, Hermetic pathworking, and "
        "Western mystery school practices, blended with Eastern "
        "mindfulness techniques brought to the West in the 19th and "
        "20th centuries."
    ),
    "tarot": (
        "The tarot originated as a card game in 15th-century northern "
        "Italy (tarocchi). Its use for divination emerged in the 18th "
        "century through the work of Jean-Baptiste Alliette (Etteilla) "
        "and later Antoine Court de Gebelin, culminating in the "
        "Rider-Waite-Smith deck of 1909 that remains the standard today."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# DEPTH LAYER PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

DEPTH_LAYERS: dict[QueryType, str] = {
    QueryType.SPELL_REQUEST: (
        "Consider the ethical implications, timing alignment, and "
        "personal connection to materials. A spell is most powerful "
        "when every element resonates with your unique energy."
    ),
    QueryType.DIVINATION_QUESTION: (
        "Approach with an open mind. Consider what you truly need "
        "to know vs. what you want to hear. Divination illuminates "
        "possibilities, not certainties."
    ),
    QueryType.HERB_CRYSTAL_QUERY: (
        "Consider sourcing ethics, personal allergies, and the "
        "difference between magical and medicinal use. Build a "
        "relationship with each material through observation and respect."
    ),
    QueryType.SABBAT_PLANNING: (
        "Honour the sabbat in a way that feels authentic to your path. "
        "Blend traditional elements with personal meaning. The Wheel "
        "turns for everyone, but each practitioner walks it differently."
    ),
    QueryType.SHADOW_WORK: (
        "This is deep, transformative work. Go at your own pace and "
        "practice self-compassion throughout. Shadow work is not about "
        "fixing what is broken — it is about integrating what has been "
        "hidden so you can become whole."
    ),
    QueryType.MEDITATION_REQUEST: (
        "Create a comfortable, safe space before beginning. There is "
        "no right or wrong experience in meditation — even restlessness "
        "is information. Trust what arises."
    ),
    QueryType.MOON_QUERY: (
        "The moon affects each person differently. Track your own "
        "energy patterns across several cycles to discover your unique "
        "lunar rhythm, rather than relying solely on general guidance."
    ),
    QueryType.TAROT_QUERY: (
        "Remember that you are the true oracle — the cards are a "
        "mirror. Your intuitive response to each image matters more "
        "than any book meaning. Read with your whole self."
    ),
    QueryType.GENERAL_WITCHCRAFT: (
        "Witchcraft is a living, evolving practice. Honour tradition "
        "while making space for your own intuition and experience. "
        "The most powerful magick comes from authenticity."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# INTENTION-EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_PREFIX_STRIPS = [
    "i want to ", "i need to ", "i would like to ", "i'd like to ",
    "how do i ", "how can i ", "can you help me with ", "help me ",
    "can you help me ", "please help me ", "i'm looking for ",
    "i am looking for ", "tell me about ", "show me ",
    "what is the best way to ", "what's the best ",
]

_GOAL_WORDS = [
    "protect", "heal", "love", "prosper", "banish", "cleanse",
    "divine", "attract", "manifest", "ground", "transform",
    "courage", "wisdom", "peace", "confidence", "create",
    "communicate", "purify", "bind", "release", "grow",
]


# ══════════════════════════════════════════════════════════════════════════════
# MYSTIC ENHANCER
# ══════════════════════════════════════════════════════════════════════════════

class MysticEnhancer:
    """Auto-detects query type and enriches every spiritual query.

    Layers applied to each query:
        1. Knowledge injection (correspondences, herbs, crystals, etc.)
        2. Current moon phase context
        3. Seasonal / Wheel of the Year context
        4. Historical / traditional context
        5. Depth / nuance guidance layer
        6. Personalization from CodexAdvisor (when user has logged sessions)

    Usage::

        enhancer = MysticEnhancer()
        result = enhancer.enhance("I want to cast a protection spell with rosemary")
        print(result.enhanced_query)
        print(f"Score: {result.score_before} -> {result.score_after}")
    """

    def __init__(self, codex_advisor=None):
        self.advisor = codex_advisor

    # ── Public API ────────────────────────────────────────────────────────

    def enhance(self, query: str) -> EnhancedQuery:
        """Enhance a spiritual query with traditional knowledge context.

        Args:
            query: The raw user query string.

        Returns:
            An :class:`EnhancedQuery` dataclass with the original query,
            enhanced query, detected type, scores, and all injected context.
        """
        score_before = self.score_query(query)

        query_type = self.detect_query_type(query)

        # Layer 1 — knowledge injection
        enhanced, knowledge_injection = self._inject_knowledge(query, query_type)

        # Layer 2 — moon context
        enhanced, moon_ctx = self._add_moon_context(enhanced)

        # Layer 3 — seasonal context
        enhanced, seasonal_ctx = self._add_seasonal_context(enhanced)

        # Layer 4 — historical context
        enhanced, historical_ctx = self._add_historical_context(
            enhanced, query_type
        )

        # Layer 5 — depth / nuance
        enhanced = self._add_depth_layer(enhanced, query_type)

        # Layer 6 — personalization from CodexAdvisor (only when user has data)
        personal_ctx = ""
        if self.advisor:
            enhanced, personal_ctx = self._add_personalization(enhanced, query)

        # Apply voice profile substitutions
        enhanced = apply_voice(enhanced)

        score_after = self.score_query(enhanced)

        # Collect injection labels for traceability
        injections: list[str] = []
        if knowledge_injection:
            injections.append("knowledge")
        if moon_ctx:
            injections.append("moon_phase")
        if seasonal_ctx:
            injections.append("seasonal")
        if historical_ctx:
            injections.append("historical")
        injections.append("depth_layer")
        if personal_ctx:
            injections.append("personalization")

        return EnhancedQuery(
            original_query=query,
            enhanced_query=enhanced,
            query_type=query_type,
            score_before=score_before,
            score_after=score_after,
            improvement=round(score_after - score_before, 1),
            injections=injections,
            moon_context=moon_ctx,
            seasonal_context=seasonal_ctx,
            historical_context=historical_ctx,
        )

    # ── Query Type Detection ──────────────────────────────────────────────

    def detect_query_type(self, query: str) -> QueryType:
        """Scan the query against QUERY_PATTERNS and return the best match.

        Counts keyword hits per type and returns the one with the most
        matches.  Falls back to ``QueryType.GENERAL_WITCHCRAFT`` when no
        keywords match.

        Args:
            query: The raw query string.

        Returns:
            The detected :class:`QueryType`.
        """
        query_lower = query.lower()
        best_type = QueryType.GENERAL_WITCHCRAFT
        best_hits = 0

        for q_type, keywords in QUERY_PATTERNS.items():
            hits = sum(1 for kw in keywords if kw in query_lower)
            if hits > best_hits:
                best_hits = hits
                best_type = q_type

        return best_type

    # ── Query Scoring ─────────────────────────────────────────────────────

    def score_query(self, query: str) -> float:
        """Score a query from 0-100 based on richness and specificity.

        Criteria (max 100):
            - **Length**: >20 chars = 10, >50 = +10, >100 = +10  (max 30)
            - **Specificity**: mentions specific herbs/crystals/colors = +5
              each  (max 20)
            - **Intention clarity**: contains clear goal words = +15
            - **Context**: mentions timing/moon/season = +10
            - **Depth**: mentions tradition, history, or multiple elements
              = +15
            - **Personalization**: mentions personal experience or
              preference = +10

        Args:
            query: The query string to score.

        Returns:
            A float between 0.0 and 100.0.
        """
        score = 0.0
        q = query.lower()
        length = len(query)

        # --- Length ---
        if length > 20:
            score += 10.0
        if length > 50:
            score += 10.0
        if length > 100:
            score += 10.0
        # Cap length contribution at 30
        length_score = min(score, 30.0)
        score = length_score

        # --- Specificity (specific herbs / crystals / colors) ---
        specificity = 0.0
        for name in HERBS:
            if name in q:
                specificity += 5.0
                if specificity >= 20.0:
                    break
        if specificity < 20.0:
            for name in CRYSTALS:
                if name in q:
                    specificity += 5.0
                    if specificity >= 20.0:
                        break
        if specificity < 20.0:
            color_words = [
                "red", "blue", "green", "black", "white", "purple",
                "pink", "gold", "silver", "orange", "yellow", "brown",
            ]
            for c in color_words:
                if c in q:
                    specificity += 5.0
                    if specificity >= 20.0:
                        break
        score += min(specificity, 20.0)

        # --- Intention clarity ---
        if any(goal in q for goal in _GOAL_WORDS):
            score += 15.0

        # --- Context (timing / moon / season) ---
        context_words = [
            "moon", "lunar", "full moon", "new moon", "waxing", "waning",
            "season", "spring", "summer", "autumn", "fall", "winter",
            "sabbat", "samhain", "yule", "imbolc", "ostara", "beltane",
            "litha", "lughnasadh", "mabon", "monday", "tuesday",
            "wednesday", "thursday", "friday", "saturday", "sunday",
        ]
        if any(cw in q for cw in context_words):
            score += 10.0

        # --- Depth (tradition / history / multiple elements) ---
        depth_words = [
            "tradition", "traditional", "history", "historical", "ancient",
            "folk", "ancestral", "celtic", "norse", "egyptian", "greek",
            "element", "fire", "water", "earth", "air", "spirit",
            "correspondence", "chakra", "planet",
        ]
        depth_hits = sum(1 for dw in depth_words if dw in q)
        if depth_hits >= 3:
            score += 15.0
        elif depth_hits >= 1:
            score += 8.0

        # --- Personalization ---
        personal_words = [
            "i feel", "my experience", "i prefer", "personally",
            "for me", "my practice", "i usually", "in my", "i have been",
            "my intention", "i want to", "i need",
        ]
        if any(pw in q for pw in personal_words):
            score += 10.0

        return min(round(score, 1), 100.0)

    # ── Layer 1: Knowledge Injection ──────────────────────────────────────

    def _inject_knowledge(
        self, query: str, query_type: QueryType
    ) -> tuple[str, str]:
        """Build and append knowledge context based on query type.

        Extracts key terms from the query and looks them up in the
        correspondences knowledge base to populate the appropriate
        injection template.

        Returns:
            A tuple of ``(enhanced_query, injection_text)``.
        """
        template = KNOWLEDGE_INJECTIONS.get(
            query_type, KNOWLEDGE_INJECTIONS[QueryType.GENERAL_WITCHCRAFT]
        )

        intention = self._extract_intention(query)
        subjects = self._extract_subjects(query)
        corr = get_correspondences_for_intention(intention)

        # Build template variables
        herbs_list = corr.get("herbs", [])[:5]
        crystals_list = corr.get("crystals", [])[:5]
        colors_list = corr.get("colors", [])[:4]
        moon_phases_list = corr.get("moon_phases", [])
        days_list = corr.get("days", [])

        timing_parts = []
        if moon_phases_list:
            timing_parts.append(", ".join(moon_phases_list[:2]))
        if days_list:
            timing_parts.append(", ".join(days_list[:2]))
        timing_str = "; ".join(timing_parts) if timing_parts else "consult the moon calendar"

        # Subject-specific lookups for herb/crystal queries
        subject_str = ", ".join(subjects) if subjects else intention
        properties_str = ""
        element_str = ""
        planet_str = ""
        pairs_str = ""
        safety_str = ""

        if subjects:
            first_subject = subjects[0]
            herb_data = HERBS.get(first_subject)
            crystal_data = CRYSTALS.get(first_subject)
            item_data = herb_data or crystal_data
            if item_data:
                properties_str = ", ".join(
                    item_data.get("magical_properties", [])[:5]
                )
                element_str = item_data.get("element", "")
                planet_str = item_data.get("planet", "")
                pairs_str = ", ".join(item_data.get("pairs_with", [])[:3])
                safety_notes = item_data.get("safety_notes", [])
                safety_str = "; ".join(safety_notes[:2]) if safety_notes else "No specific cautions"

        # Safety notes aggregation for spell requests
        safety_notes_combined: list[str] = []
        for herb_name in herbs_list[:3]:
            hd = HERBS.get(herb_name, {})
            notes = hd.get("safety_notes", [])
            if notes and notes[0] != "Generally safe":
                safety_notes_combined.append(f"{herb_name}: {notes[0]}")

        # Sabbat-specific lookups
        sabbat_str = ""
        themes_str = ""
        activities_str = ""
        sabbat_colors = ""
        sabbat_herbs = ""
        _sabbat_names = [
            "samhain", "yule", "imbolc", "ostara",
            "beltane", "litha", "lughnasadh", "mabon",
        ]
        for sn in _sabbat_names:
            if sn in query.lower():
                sabbat_str = sn.title()
                from grimoire.knowledge.wheel_of_year import SABBATS
                sdata = SABBATS.get(sn, {})
                if sdata:
                    themes_str = ", ".join(sdata.get("themes", [])[:5])
                    rituals = sdata.get("rituals", [])
                    activities_str = "; ".join(rituals[:3]) if rituals else ""
                    sc = sdata.get("correspondences", {})
                    sabbat_colors = ", ".join(sc.get("colors", [])[:4])
                    sabbat_herbs = ", ".join(sc.get("herbs", [])[:4])
                break

        # Tarot-specific lookups
        related_cards = get_cards_for_intention(intention)
        card_names = [c.get("name", "") for c in related_cards[:3]]
        cards_str = ", ".join(card_names) if card_names else "draw intuitively"

        # Divination-specific
        divination_methods = {
            "tarot": "Tarot", "scry": "Scrying", "rune": "Rune casting",
            "pendulum": "Pendulum", "oracle": "Oracle cards",
            "i ching": "I Ching",
        }
        method_str = "Tarot"
        for kw, method in divination_methods.items():
            if kw in query.lower():
                method_str = method
                break

        # Meditation-specific
        from grimoire.knowledge.correspondences import ELEMENTS
        element_focus = "spirit"
        time_of_day = "dawn or dusk"
        for el_key, el_data in ELEMENTS.items():
            if el_key in query.lower():
                element_focus = el_data.get("name", el_key)
                time_of_day = el_data.get("time_of_day", "the liminal hour")
                break

        # Spread recommendation for tarot
        spread_str = "three-card Past/Present/Future"
        if "celtic" in query.lower():
            spread_str = "Celtic Cross (10 cards)"
        elif "shadow" in query.lower():
            spread_str = "Shadow Work (5 cards)"
        elif "relationship" in query.lower():
            spread_str = "Relationship (7 cards)"
        elif "decision" in query.lower():
            spread_str = "Two Paths (5 cards)"
        elif "full moon" in query.lower():
            spread_str = "Full Moon (5 cards)"
        elif "new moon" in query.lower():
            spread_str = "New Moon (6 cards)"

        # Get current moon for moon queries
        now = datetime.datetime.now()
        phase_key, illumination = calculate_moon_phase(
            now.year, now.month, now.day
        )
        phase_data = get_phase_data(phase_key) or {}
        phase_name = phase_data.get("name", phase_key.replace("_", " ").title())
        moon_energy = phase_data.get("magical_energy", "")
        moon_best_for = ", ".join(phase_data.get("best_for", [])[:4])
        moon_avoid = ", ".join(phase_data.get("avoid", [])[:3])

        # Format the injection using safe .format() with defaults
        try:
            injection = template.format(
                intention=intention or "your intention",
                herbs=", ".join(herbs_list) if herbs_list else "consult your garden",
                crystals=", ".join(crystals_list) if crystals_list else "clear quartz (all-purpose)",
                colors=", ".join(colors_list) if colors_list else "white (universal)",
                timing=timing_str,
                safety_notes=(
                    "Safety: " + "; ".join(safety_notes_combined)
                    if safety_notes_combined
                    else ""
                ),
                subject=subject_str or "the subject in question",
                properties=properties_str or "various magical uses",
                element=element_str or "varies",
                planet=planet_str or "varies",
                pairs=pairs_str or "complementary materials",
                safety=safety_str or "research before ingesting any herb",
                sabbat=sabbat_str or "this sabbat",
                themes=themes_str or "seasonal themes",
                activities=activities_str or "traditional observances",
                phase=phase_name,
                sign="the current sign",
                energy=moon_energy or "current lunar energy",
                best_for=moon_best_for or "various workings",
                avoid=moon_avoid or "nothing specific",
                method=method_str,
                focus=intention or "your question",
                symbols=cards_str,
                cards=cards_str,
                spread=spread_str,
                theme=intention or "self-exploration",
                time_of_day=time_of_day,
            )
        except KeyError:
            # Fallback if any template key is missing
            injection = (
                f"Relevant correspondences for '{intention}': "
                f"Herbs: {', '.join(herbs_list[:3]) or 'varies'}. "
                f"Crystals: {', '.join(crystals_list[:3]) or 'varies'}."
            )

        # Override some injections with richer sabbat-specific data
        if query_type == QueryType.SABBAT_PLANNING and sabbat_str:
            injection = KNOWLEDGE_INJECTIONS[QueryType.SABBAT_PLANNING].format(
                sabbat=sabbat_str,
                colors=sabbat_colors or ", ".join(colors_list),
                herbs=sabbat_herbs or ", ".join(herbs_list),
                themes=themes_str or "seasonal celebration",
                activities=activities_str or "traditional observances",
            )

        enhanced = f"{query}\n\n[Knowledge Context] {injection}"
        return enhanced, injection

    # ── Layer 2: Moon Context ─────────────────────────────────────────────

    def _add_moon_context(self, query: str) -> tuple[str, str]:
        """Append current moon phase information to the query.

        Returns:
            A tuple of ``(enhanced_query, moon_context_string)``.
        """
        now = datetime.datetime.now()
        phase_key, illumination = calculate_moon_phase(
            now.year, now.month, now.day
        )
        phase_data = get_phase_data(phase_key)
        if not phase_data:
            return query, ""

        phase_name = phase_data.get("name", phase_key)
        energy = phase_data.get("magical_energy", "")
        best_for = phase_data.get("best_for", [])[:4]
        daily_guidance = phase_data.get("daily_guidance", "")

        moon_text = (
            f"Current Moon: {phase_name} "
            f"({round(illumination * 100)}% illuminated). "
            f"Energy: {energy}. "
            f"Best for: {', '.join(best_for)}. "
            f"Guidance: {daily_guidance}"
        )

        enhanced = f"{query}\n\n[Moon Context] {moon_text}"
        return enhanced, moon_text

    # ── Layer 3: Seasonal Context ─────────────────────────────────────────

    def _add_seasonal_context(self, query: str) -> tuple[str, str]:
        """Append seasonal / Wheel of the Year context to the query.

        Returns:
            A tuple of ``(enhanced_query, seasonal_context_string)``.
        """
        now = datetime.datetime.now()
        seasonal_text = get_seasonal_context(now.month)

        # Add next sabbat info
        sabbat_name, sabbat_data, days_until = get_next_sabbat(
            now.month, now.day
        )
        next_sabbat_str = (
            f" Next sabbat: {sabbat_name} in {days_until} days."
        )

        full_seasonal = f"{seasonal_text}{next_sabbat_str}"
        enhanced = f"{query}\n\n[Seasonal Context] {full_seasonal}"
        return enhanced, full_seasonal

    # ── Layer 4: Historical Context ───────────────────────────────────────

    def _add_historical_context(
        self, query: str, query_type: QueryType
    ) -> tuple[str, str]:
        """Add a sentence of traditional/historical context.

        Selects from ``HISTORICAL_NOTES`` based on query type and
        specific keywords found in the query.

        Returns:
            A tuple of ``(enhanced_query, historical_context_string)``.
        """
        # Map query types to historical note keys
        type_to_history: dict[QueryType, str] = {
            QueryType.SPELL_REQUEST: "candle_magick",
            QueryType.DIVINATION_QUESTION: "divination",
            QueryType.HERB_CRYSTAL_QUERY: "herb_craft",
            QueryType.SABBAT_PLANNING: "sabbats",
            QueryType.SHADOW_WORK: "shadow_work",
            QueryType.MEDITATION_REQUEST: "meditation",
            QueryType.MOON_QUERY: "moon_magick",
            QueryType.TAROT_QUERY: "tarot",
            QueryType.GENERAL_WITCHCRAFT: "herb_craft",
        }

        # Refine based on specific query content
        q_lower = query.lower()
        history_key = type_to_history.get(query_type, "herb_craft")

        # Override with more specific matches
        if "candle" in q_lower:
            history_key = "candle_magick"
        elif "crystal" in q_lower or "stone" in q_lower:
            history_key = "crystal_work"
        elif "herb" in q_lower or "plant" in q_lower:
            history_key = "herb_craft"
        elif "tarot" in q_lower or "card" in q_lower:
            history_key = "tarot"
        elif any(s in q_lower for s in [
            "samhain", "yule", "imbolc", "ostara",
            "beltane", "litha", "lughnasadh", "mabon",
            "sabbat", "wheel of year",
        ]):
            history_key = "sabbats"

        historical_text = HISTORICAL_NOTES.get(history_key, "")
        if not historical_text:
            return query, ""

        enhanced = f"{query}\n\n[Historical Context] {historical_text}"
        return enhanced, historical_text

    # ── Layer 5: Depth Layer ──────────────────────────────────────────────

    def _add_depth_layer(self, query: str, query_type: QueryType) -> str:
        """Add nuance prompting appropriate to the query type.

        Args:
            query: The query built so far.
            query_type: The detected query type.

        Returns:
            The query with a depth/nuance layer appended.
        """
        depth_text = DEPTH_LAYERS.get(
            query_type, DEPTH_LAYERS[QueryType.GENERAL_WITCHCRAFT]
        )
        return f"{query}\n\n[Depth] {depth_text}"

    # ── Layer 6: Personalization ──────────────────────────────────────────

    def _add_personalization(self, query: str, raw_query: str) -> tuple[str, str]:
        """Inject personalization context from the practitioner's history.

        Only adds context when the CodexAdvisor has meaningful data
        (session_count > 0). Returns empty string when no data exists,
        so brand-new users see no change.

        Returns:
            A tuple of ``(enhanced_query, personalization_text)``.
        """
        if not self.advisor:
            return query, ""

        try:
            intention = self._extract_intention(raw_query)
            ctx = self.advisor.get_personalization_context(intention)
        except Exception:
            return query, ""

        if ctx.get("session_count", 0) == 0:
            return query, ""

        parts: list[str] = []

        # Session and streak info
        parts.append(
            f"Practitioner has logged {ctx['session_count']} sessions"
        )
        if ctx.get("streak", 0) > 1:
            parts.append(f"with a {ctx['streak']}-day streak")

        # Preferred materials
        if ctx.get("preferred_herbs"):
            parts.append(
                f"Preferred herbs: {', '.join(ctx['preferred_herbs'][:3])}"
            )
        if ctx.get("preferred_crystals"):
            parts.append(
                f"Preferred crystals: {', '.join(ctx['preferred_crystals'][:3])}"
            )

        # Discovery suggestion
        if ctx.get("discovery_herb"):
            parts.append(f"New herb to explore: {ctx['discovery_herb']}")
        if ctx.get("discovery_crystal"):
            parts.append(f"New crystal to explore: {ctx['discovery_crystal']}")

        # Best moon phase
        if ctx.get("best_moon_phase"):
            parts.append(
                f"Most effective moon phase: {ctx['best_moon_phase']}"
            )

        if not parts:
            return query, ""

        personal_text = ". ".join(parts) + "."
        enhanced = f"{query}\n\n[Personalization] {personal_text}"
        return enhanced, personal_text

    # ── Extraction Helpers ────────────────────────────────────────────────

    def _extract_intention(self, query: str) -> str:
        """Extract the core intention from a query string.

        Strips common conversational prefixes like "I want to",
        "How do I", "Can you help me with" to isolate the actual
        magical intention.

        Args:
            query: The raw query string.

        Returns:
            A cleaned intention string.
        """
        result = query.strip()
        result_lower = result.lower()

        for prefix in _PREFIX_STRIPS:
            if result_lower.startswith(prefix):
                result = result[len(prefix):]
                result_lower = result.lower()
                break

        # Strip trailing punctuation
        result = result.rstrip("?.!,;:")

        # If we stripped everything, return the original
        if len(result.strip()) < 3:
            return query.strip()

        return result.strip()

    def _extract_subjects(self, query: str) -> list[str]:
        """Extract specific herbs, crystals, or other subjects from the query.

        Checks every word/phrase in the query against the HERBS and
        CRYSTALS dictionaries.

        Args:
            query: The raw query string.

        Returns:
            A deduplicated list of matched subject keys.
        """
        q_lower = query.lower()
        found: list[str] = []
        seen: set[str] = set()

        # Check multi-word keys first (e.g. "rose quartz", "black tourmaline")
        all_items = list(HERBS.keys()) + list(CRYSTALS.keys())
        # Sort by length descending to match longer names first
        all_items.sort(key=len, reverse=True)

        for name in all_items:
            if name in q_lower and name not in seen:
                seen.add(name)
                found.append(name)

        return found
