"""
SpellScout -- Intention Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Part of the FORGE intelligence layer for the Grimoire Intelligence System.

Takes any free-form intention string ("I want protection", "love spell",
"attract money for a new home") and returns a rich analysis: detected
category, full correspondences, completeness score, identified gaps,
a beginner-friendly quick-start guide, and actionable enhancement tips.

The voice is warm and encouraging -- never clinical. We want practitioners
to feel welcomed, not judged.
"""

from __future__ import annotations

import random
from typing import Any

from grimoire.models import SpellScoutResult, IntentionCategory, Correspondence
from grimoire.knowledge.correspondences import (
    HERBS,
    CRYSTALS,
    COLORS,
    ELEMENTS,
    PLANETS,
    DAYS_OF_WEEK,
    INTENTION_MAP,
    get_correspondences_for_intention,
)
from grimoire.knowledge.numerology import get_magical_number
from grimoire.voice import (
    get_opening,
    get_closing,
    get_encouragement,
    apply_voice,
    VOICE_PROFILE,
)


# ---------------------------------------------------------------------------
# Keyword map -- each IntentionCategory value maps to trigger phrases that
# help us detect the practitioner's true intention even from conversational
# or poetic phrasing.
# ---------------------------------------------------------------------------

INTENTION_KEYWORDS: dict[str, list[str]] = {
    "protection": [
        "protect", "ward", "shield", "safe", "guard", "defend", "banish negative",
    ],
    "love": [
        "love", "romance", "attract partner", "relationship", "heart",
        "soulmate", "self-love",
    ],
    "prosperity": [
        "money", "wealth", "prosper", "abundance", "financial", "job",
        "career", "success", "rich",
    ],
    "healing": [
        "heal", "health", "recovery", "cure", "wellness", "pain",
        "sickness", "restore",
    ],
    "divination": [
        "divine", "divination", "future", "scry", "psychic", "tarot",
        "oracle", "insight",
    ],
    "banishing": [
        "banish", "remove", "get rid", "repel", "exorcise",
        "break curse", "undo",
    ],
    "cleansing": [
        "cleanse", "purify", "clear", "purge", "renew", "refresh", "clean",
    ],
    "creativity": [
        "creat", "inspir", "art", "muse", "imagination", "innovat", "express",
    ],
    "wisdom": [
        "wisdom", "knowledge", "learn", "study", "understand", "enlighten",
        "truth",
    ],
    "confidence": [
        "confiden", "courage", "brave", "bold", "self-esteem", "empower",
        "strength",
    ],
    "communication": [
        "communicat", "speak", "voice", "express", "listen", "understand",
        "eloquen",
    ],
    "grounding": [
        "ground", "center", "root", "stable", "anchor", "calm", "balance",
    ],
    "transformation": [
        "transform", "change", "evolve", "metamorphos", "rebirth",
        "transition",
    ],
    "peace": [
        "peace", "calm", "tranquil", "seren", "harmony", "relax", "still",
    ],
    "courage": [
        "courage", "brave", "fearless", "bold", "warrior", "strength",
        "valor",
    ],
}

# All correspondence categories we consider for completeness scoring.
_ALL_CATEGORIES = [
    "herbs", "crystals", "colors", "elements", "planets",
    "days", "numbers", "deities", "incense",
]


# ---------------------------------------------------------------------------
# Quick-start templates -- these feel like a friendly mentor talking you
# through your very first step. We fill in specifics at runtime.
# ---------------------------------------------------------------------------

_QUICK_START_TEMPLATES = [
    (
        "A wonderful place to begin your {category} working is with a simple "
        "candle ritual. Light a {color} candle, hold a piece of {crystal} in "
        "your hand, and speak your intention aloud. {herb_tip} "
        "{closing}"
    ),
    (
        "Start small and trust the process. Gather {herb} and {crystal}, "
        "place them on a clean surface, and sit quietly with your intention "
        "for a few moments. {herb_tip} "
        "{closing}"
    ),
    (
        "Your {category} journey can start tonight. Brew a cup of {herb} tea "
        "(if the herb is safe to drink), hold {crystal} in your palm, and "
        "visualize your intention wrapping around you like warm light. "
        "{herb_tip} {closing}"
    ),
    (
        "For a gentle beginning, try carrying {crystal} in your pocket on a "
        "{day} and keeping a sprig of {herb} somewhere you will see it daily. "
        "Each time you notice it, silently affirm your intention. "
        "{herb_tip} {closing}"
    ),
]

# Friendly transition phrases for enhancement suggestions.
_ENHANCEMENT_OPENERS = [
    "To deepen your practice, consider adding",
    "You could strengthen this working by including",
    "A lovely addition would be",
    "When you feel ready, try weaving in",
    "For an extra layer of intention, bring in",
]


# ============================================================================
# SpellScout
# ============================================================================

class SpellScout:
    """Intention analysis and correspondence recommendation engine.

    The SpellScout reads any intention string -- whether it is a single word
    like "protection" or a heartfelt paragraph -- and returns everything a
    practitioner needs to get started: which magical category it aligns with,
    the herbs, crystals, colors, timing, and numbers that support it, a
    completeness score, a list of gaps, and a warm quick-start paragraph.

    Usage::

        scout = SpellScout()
        result = scout.analyze("I want to attract love and heal my heart")

        print(result.category)            # IntentionCategory.LOVE
        print(result.completeness_score)  # 78.0
        print(result.quick_start)         # A warm paragraph ...
        print(result.suggestions)         # ["Consider adding ...", ...]
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, intention: str) -> SpellScoutResult:
        """Run full intention analysis and return a rich result.

        Steps:
            1. Detect the most likely IntentionCategory.
            2. Pull all matching correspondences for that category.
            3. Score how *complete* those correspondences are (0-100).
            4. Identify which correspondence categories are missing (gaps).
            5. Generate a beginner-friendly quick-start paragraph.
            6. Generate 3-5 actionable enhancement suggestions.

        Args:
            intention: A free-form string describing what the practitioner
                wants to achieve. Examples: "I want protection", "love spell",
                "attract money".

        Returns:
            A fully populated SpellScoutResult dataclass.
        """
        category = self.detect_category(intention)
        correspondences = self.get_correspondences(category)
        completeness = self.score_completeness(correspondences)
        gaps = self._identify_gaps(correspondences)
        quick_start = self.suggest_quick_start(intention, category, correspondences)
        suggestions = self.suggest_enhancements(category, correspondences)

        return SpellScoutResult(
            intention=intention,
            category=category,
            correspondences=correspondences,
            completeness_score=completeness,
            gaps=gaps,
            quick_start=quick_start,
            suggestions=suggestions,
        )

    def detect_category(self, intention: str) -> IntentionCategory:
        """Detect the single best-matching IntentionCategory.

        The algorithm scans the intention text for each category's keyword
        list and counts how many keywords appear. The category with the most
        hits wins. Ties are broken by the order categories appear in
        ``INTENTION_KEYWORDS`` (i.e., the first match wins).

        If nothing matches at all we default to ``IntentionCategory.PROTECTION``
        -- a safe, broadly useful fallback.

        Args:
            intention: Free-form intention string.

        Returns:
            The detected IntentionCategory enum member.
        """
        intention_lower = intention.lower()

        best_category: str | None = None
        best_hits = 0

        for category_value, keywords in INTENTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in intention_lower)
            if hits > best_hits:
                best_hits = hits
                best_category = category_value

        if best_category is None:
            # Graceful fallback -- protection is a wonderful starting point
            return IntentionCategory.PROTECTION

        return IntentionCategory(best_category)

    def get_correspondences(self, category: IntentionCategory) -> dict[str, list[str]]:
        """Return full correspondences for an IntentionCategory.

        Pulls the core data from ``INTENTION_MAP`` and enriches it with
        number, deity, and incense information derived from the broader
        knowledge base.

        Keys in the returned dict:
            herbs, crystals, colors, elements, planets, days, numbers,
            deities, incense.

        Args:
            category: An IntentionCategory enum member.

        Returns:
            A dict whose values are lists of strings. Missing categories
            will have empty lists.
        """
        cat_key = category.value
        base = INTENTION_MAP.get(cat_key, {})

        # Start with the core lists from INTENTION_MAP.
        herbs = list(base.get("herbs", []))
        crystals = list(base.get("crystals", []))
        colors = list(base.get("colors", []))
        elements = [base["element"]] if base.get("element") else []
        planets = [base["planet"]] if base.get("planet") else []
        days = [base["day"]] if base.get("day") else []

        # Enrich: numerological number for this category.
        num_info = get_magical_number(cat_key)
        numbers = [str(num_info.get("number", ""))] if num_info else []

        # Enrich: deities -- gather from the first 3 herbs and first 3
        # crystals that have deity info.
        deities = self._collect_deities(herbs[:3], crystals[:3])

        # Enrich: incense -- herbs commonly burned as incense.
        incense_herbs = {"frankincense", "myrrh", "sandalwood", "sage", "cedar",
                         "juniper", "pine", "patchouli", "lavender", "rosemary",
                         "wormwood", "star anise", "cinnamon", "clove"}
        incense = [h for h in herbs if h in incense_herbs]

        return {
            "herbs": herbs,
            "crystals": crystals,
            "colors": colors,
            "elements": elements,
            "planets": planets,
            "days": days,
            "numbers": numbers,
            "deities": deities,
            "incense": incense,
        }

    def score_completeness(self, correspondences: dict[str, list[str]]) -> float:
        """Score how complete a set of correspondences is on a 0-100 scale.

        Each of the nine standard categories (herbs, crystals, colors,
        elements, planets, days, numbers, deities, incense) is worth up to
        ~11.1 points. A category with at least one entry earns its full
        share; an empty category earns nothing.

        This is intentionally generous -- a practitioner who has *something*
        in every bucket is well-prepared, even if the lists are short.

        Args:
            correspondences: The dict returned by ``get_correspondences()``.

        Returns:
            A float from 0.0 to 100.0.
        """
        if not correspondences:
            return 0.0

        filled = sum(
            1 for cat in _ALL_CATEGORIES
            if correspondences.get(cat) and len(correspondences[cat]) > 0
        )
        total = len(_ALL_CATEGORIES)
        return round((filled / total) * 100, 1)

    def suggest_quick_start(
        self,
        intention: str,
        category: IntentionCategory,
        correspondences: dict[str, list[str]],
    ) -> str:
        """Generate a warm, beginner-friendly quick-start paragraph.

        Selects a template and fills it with actual correspondences from
        the analysis, then applies the Grimoire voice profile.

        Args:
            intention: The original intention string.
            category: The detected IntentionCategory.
            correspondences: The full correspondences dict.

        Returns:
            A 3-5 sentence paragraph ready to show a practitioner.
        """
        # Pick concrete items to mention.
        herb = self._pick_one(correspondences.get("herbs", []), "rosemary")
        crystal = self._pick_one(correspondences.get("crystals", []), "clear quartz")
        color = self._pick_one(correspondences.get("colors", []), "white")
        day = self._pick_one(correspondences.get("days", []), "sunday")

        # Look up a beginner tip from the herb knowledge base.
        from grimoire.knowledge.correspondences import HERBS as _HERBS
        herb_data = _HERBS.get(herb.lower(), {})
        herb_tip = herb_data.get("beginner_tip", "")
        if herb_tip:
            herb_tip = f"A small tip: {herb_tip}"

        # Pick a random template and fill it in.
        template = random.choice(_QUICK_START_TEMPLATES)
        closing = get_closing()
        category_name = category.value.replace("_", " ")

        quick_start = template.format(
            category=category_name,
            herb=herb.title(),
            crystal=crystal.title(),
            color=color,
            day=day.title(),
            herb_tip=herb_tip,
            closing=closing,
        )

        # Prepend a warm opening.
        opening = get_opening()
        encouragement = get_encouragement()

        full_text = f"{opening} {category_name} work. {quick_start} {encouragement}"

        return apply_voice(full_text.strip())

    def suggest_enhancements(
        self,
        category: IntentionCategory,
        current_correspondences: dict[str, list[str]],
    ) -> list[str]:
        """Suggest 3-5 enhancements to strengthen the practitioner's working.

        Looks at what is missing (gaps) and also suggests timing, layering,
        and deepening strategies in a friendly, non-judgmental tone.

        Args:
            category: The detected IntentionCategory.
            current_correspondences: Current correspondences dict.

        Returns:
            A list of 3-5 suggestion strings.
        """
        suggestions: list[str] = []
        gaps = self._identify_gaps(current_correspondences)
        cat_name = category.value.replace("_", " ")

        # --- Suggestion 1: Address the first gap ---
        if "deities" in gaps:
            deity_note = self._suggest_deities_for_category(category)
            if deity_note:
                suggestions.append(deity_note)

        if "incense" in gaps:
            suggestions.append(
                f"{random.choice(_ENHANCEMENT_OPENERS)} burning incense "
                f"such as frankincense or sandalwood to deepen your "
                f"{cat_name} ritual. The scent helps anchor your intention "
                f"in the physical world."
            )

        if "numbers" in gaps:
            num_info = get_magical_number(category.value)
            num = num_info.get("number", 7)
            suggestions.append(
                f"The number {num} resonates with {cat_name} energy. Try "
                f"repeating your affirmation {num} times, using {num} "
                f"knots in cord magick, or lighting {num} candles."
            )

        if "crystals" in gaps:
            suggestions.append(
                f"{random.choice(_ENHANCEMENT_OPENERS)} a crystal ally. "
                f"Even a small tumbled stone carried in your pocket can "
                f"amplify your {cat_name} work throughout the day."
            )

        if "herbs" in gaps:
            suggestions.append(
                f"{random.choice(_ENHANCEMENT_OPENERS)} herbal support. "
                f"Brewing a simple tea, tucking a sachet under your pillow, "
                f"or adding dried herbs to a spell jar are all beautiful "
                f"entry points."
            )

        # --- Timing suggestion (always helpful) ---
        day = self._pick_one(current_correspondences.get("days", []), "")
        moon = INTENTION_MAP.get(category.value, {}).get("moon_phase", "")
        if day and moon:
            suggestions.append(
                f"For the strongest alignment, perform your {cat_name} "
                f"working on a {day.title()} during the {moon}. Timing "
                f"adds a quiet but powerful layer to any practice."
            )
        elif day:
            suggestions.append(
                f"Try performing your {cat_name} working on a "
                f"{day.title()} to align with planetary energies."
            )

        # --- Layering suggestion ---
        elements = current_correspondences.get("elements", [])
        if elements:
            el = elements[0]
            el_data = ELEMENTS.get(el, {})
            direction = el_data.get("direction", "")
            if direction:
                suggestions.append(
                    f"The element of {el.title()} rules the {direction}. "
                    f"Face {direction} when you begin your working, or "
                    f"place your materials in the {direction} part of your "
                    f"altar for extra resonance."
                )

        # --- Journaling suggestion (universally good) ---
        suggestions.append(
            f"After your {cat_name} working, spend a few quiet minutes "
            f"journaling about what you felt. Tracking your experiences "
            f"helps you notice patterns and grow in confidence over time."
        )

        # Return 3-5 suggestions, trimming if we generated too many.
        return suggestions[:5] if len(suggestions) > 5 else suggestions[:max(3, len(suggestions))]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_gaps(self, correspondences: dict[str, list[str]]) -> list[str]:
        """Return a list of correspondence category names that are empty.

        Args:
            correspondences: The full correspondences dict.

        Returns:
            A list of category name strings (e.g., ["deities", "incense"]).
        """
        return [
            cat for cat in _ALL_CATEGORIES
            if not correspondences.get(cat)
        ]

    def _collect_deities(
        self,
        herbs: list[str],
        crystals: list[str],
    ) -> list[str]:
        """Gather unique deity names from herb and crystal entries.

        We look up each name in the knowledge base and collect any
        associated deities, deduplicating and capping at 5.

        Args:
            herbs: Herb names to look up.
            crystals: Crystal names to look up.

        Returns:
            A deduplicated list of deity name strings, max 5.
        """
        seen: set[str] = set()
        deities: list[str] = []

        for herb_name in herbs:
            herb_data = HERBS.get(herb_name.lower(), {})
            for d in herb_data.get("deities", []):
                if d not in seen:
                    seen.add(d)
                    deities.append(d)

        for crystal_name in crystals:
            crystal_data = CRYSTALS.get(crystal_name.lower(), {})
            for d in crystal_data.get("deities", []):
                if d not in seen:
                    seen.add(d)
                    deities.append(d)

        return deities[:5]

    def _suggest_deities_for_category(self, category: IntentionCategory) -> str:
        """Return a suggestion string for working with deities, or empty.

        Uses lightweight mappings -- no hard-coded theology, just common
        associations from eclectic and Wiccan traditions.
        """
        deity_hints: dict[str, str] = {
            "protection": (
                "You might invoke protective deities such as Hecate, Mars, "
                "or Brigid to lend their strength to your ward."
            ),
            "love": (
                "Aphrodite, Venus, and Freya are traditional allies in love "
                "workings. A small offering of rose petals honors their energy."
            ),
            "prosperity": (
                "Jupiter, Lakshmi, and Cernunnos are wonderful patrons of "
                "abundance. Consider leaving a coin on your altar as an offering."
            ),
            "healing": (
                "Brigid, Apollo, and Hygieia are healers in many traditions. "
                "Ask for their gentle guidance as you work."
            ),
            "divination": (
                "Hecate, Thoth, and Apollo are keepers of hidden knowledge. "
                "Light a candle in their honor before you begin scrying."
            ),
            "banishing": (
                "Hecate, Kali, and the Morrighan excel at clearing what no "
                "longer serves. Speak firmly and with respect when you call "
                "on them."
            ),
            "cleansing": (
                "Selene, Brigid, and Saraswati carry purifying energy. A "
                "prayer to any of them while you cleanse adds depth to the work."
            ),
            "creativity": (
                "The Muses, Brigid, and Saraswati inspire creative fire. "
                "Leave an offering of art, music, or words on your altar."
            ),
            "wisdom": (
                "Athena, Thoth, and Odin have walked the long road of "
                "wisdom. Meditate with their names to open channels of insight."
            ),
            "confidence": (
                "Sekhmet, Ares, and Freya embody fierce self-assurance. "
                "Light a gold candle and ask for their courage."
            ),
            "communication": (
                "Mercury, Hermes, and Thoth govern the spoken and written "
                "word. Invoke them before important conversations."
            ),
            "grounding": (
                "Gaia, Cernunnos, and Pachamama anchor us to the earth. "
                "Touch the ground and ask for their steadiness."
            ),
            "transformation": (
                "Persephone, Kali, and the Phoenix archetype guide "
                "transformation. Honor the death that makes rebirth possible."
            ),
            "peace": (
                "Kuan Yin, Pax, and Brigid carry deep peace. Sit with their "
                "names in meditation and let the stillness settle in."
            ),
            "courage": (
                "Ares, Athena, and the Morrighan stand with those who face "
                "their fears. Carry a red stone and know they walk beside you."
            ),
        }
        return deity_hints.get(category.value, "")

    @staticmethod
    def _pick_one(items: list[str], fallback: str) -> str:
        """Pick a random item from a list, or return the fallback."""
        if not items:
            return fallback
        return random.choice(items)
