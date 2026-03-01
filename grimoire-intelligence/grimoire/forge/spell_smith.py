"""SpellSmith — the auto-generation engine of the Grimoire FORGE layer.

Generates complete spells, rituals, meditations, journal prompts,
tarot spreads, and daily practice suggestions. ALL generation is
template-based with zero AI API calls.
"""

import random
import datetime
from pathlib import Path

from grimoire.forge.variation_engine import (
    VariationEngine,
    AFFIRMATION_POOLS,
    ELEMENT_IMAGERY_POOLS,
    AFTERCARE_POOLS,
    PREPARATION_POOLS,
    TIMING_ADVICE_POOLS,
    DAILY_SUGGESTION_POOLS,
    QUICK_PRACTICE_POOLS,
    SPELL_TITLE_POOLS,
    SPELL_DESCRIPTION_POOLS,
    RITUAL_TITLE_POOLS,
    SABBAT_RITUAL_TITLE_POOLS,
    RITUAL_ALTAR_POOLS,
    RITUAL_OPENING_POOLS,
    RITUAL_BODY_STEP_POOLS,
    RITUAL_CLOSING_POOLS,
    RITUAL_PREPARATION_POOLS,
    MEDITATION_TITLE_POOLS,
    MEDITATION_GROUNDING_POOLS,
    MEDITATION_PEAK_POOLS,
    MEDITATION_RETURN_POOLS,
)
from grimoire.models import (
    GeneratedSpell,
    GeneratedRitual,
    GeneratedMeditation,
    RitualPlan,
    DailyPractice,
    IntentionCategory,
    SpellType,
    Difficulty,
)
from grimoire.knowledge.spell_templates import (
    SPELL_TYPES,
    RITUAL_STRUCTURE,
    get_spell_template,
)
from grimoire.knowledge.meditation_frameworks import (
    MEDITATION_FRAMEWORKS,
    get_meditation,
)
from grimoire.knowledge.correspondences import (
    HERBS,
    CRYSTALS,
    COLORS,
    INTENTION_MAP,
    get_correspondences_for_intention,
)
from grimoire.knowledge.journal_prompts import (
    get_daily_prompt,
    get_moon_prompts,
    get_sabbat_prompts,
)
from grimoire.knowledge.numerology import get_magical_number
from grimoire.voice import get_opening, get_closing, get_encouragement, apply_voice


# ---------------------------------------------------------------------------
# Difficulty -> material count
# ---------------------------------------------------------------------------

_DIFFICULTY_MATERIAL_COUNT = {
    "beginner": 3,
    "intermediate": 5,
    "advanced": 7,
}

# ---------------------------------------------------------------------------
# Day-of-week rulers (for daily practice generation)
# ---------------------------------------------------------------------------

_DAY_RULERS = {
    0: ("Monday", "moon"),
    1: ("Tuesday", "mars"),
    2: ("Wednesday", "mercury"),
    3: ("Thursday", "jupiter"),
    4: ("Friday", "venus"),
    5: ("Saturday", "saturn"),
    6: ("Sunday", "sun"),
}

# ---------------------------------------------------------------------------
# Element imagery for visualizations
# ---------------------------------------------------------------------------

_ELEMENT_IMAGERY = {
    "fire": (
        "See a brilliant flame kindling before you, its warmth washing over "
        "your skin. The fire dances with purpose, each flicker shaping your "
        "intention into golden light. Feel the heat gather at your solar "
        "plexus, igniting your willpower and burning away all that stands "
        "between you and your goal."
    ),
    "water": (
        "Imagine a still, moonlit pool at your feet. As you breathe, gentle "
        "ripples spread outward, carrying your intention across the water. "
        "Feel the cool, cleansing energy wash through you — purifying, "
        "healing, flowing into every corner of your being like a sacred "
        "tide returning home."
    ),
    "earth": (
        "Visualize rich, dark soil beneath your hands. Feel its weight and "
        "warmth, teeming with life just beneath the surface. Press your "
        "intention into the earth like a seed. Feel roots extend from your "
        "body downward, anchoring you, while green shoots of manifestation "
        "reach toward the light."
    ),
    "air": (
        "A gentle breeze rises around you, carrying the scent of herbs and "
        "distant rain. Speak your intention into the wind and watch the "
        "words become silver threads spiraling upward. Feel clarity flood "
        "your mind as the air sharpens your focus and lifts your spirit "
        "like wings unfolding."
    ),
    "spirit": (
        "A column of luminous white light descends from above, passing "
        "through your crown and filling your entire being. You stand at "
        "the crossroads of all elements, all directions, all possibilities. "
        "Your intention resonates outward in every direction at once, "
        "connecting you to the web of all magick."
    ),
}

# ---------------------------------------------------------------------------
# Affirmation templates — one per IntentionCategory
# ---------------------------------------------------------------------------

_AFFIRMATIONS = {
    "protection": (
        "I am surrounded by an unbreakable shield of light. Nothing that "
        "is not for my highest good may enter."
    ),
    "love": (
        "I am worthy of deep, abundant love. My heart is open and magnetic."
    ),
    "prosperity": (
        "Abundance flows to me from expected and unexpected sources. "
        "I am a magnet for prosperity."
    ),
    "healing": (
        "My body, mind, and spirit align in radiant health. Healing "
        "energy flows through me with every breath."
    ),
    "divination": (
        "My inner sight is clear and true. I trust the messages the "
        "universe reveals to me."
    ),
    "banishing": (
        "I release all that no longer serves me. Negativity dissolves "
        "in the light of my will."
    ),
    "cleansing": (
        "I am purified in body, mind, and spirit. Only what is clean "
        "and true remains within my space."
    ),
    "creativity": (
        "Creative energy flows through me freely and joyfully. I am "
        "an open channel for inspiration."
    ),
    "wisdom": (
        "Ancient wisdom lives within me. I access deeper knowing with "
        "every still moment I create."
    ),
    "confidence": (
        "I stand in my power, radiant and unshakable. I trust myself "
        "completely and act with bold certainty."
    ),
    "communication": (
        "My words carry truth and clarity. I speak with confidence and "
        "listen with compassion."
    ),
    "grounding": (
        "I am rooted deep in the earth, stable and strong. No storm "
        "can shake my foundation."
    ),
    "transformation": (
        "I welcome change as a sacred teacher. I release the old self "
        "and step into who I am becoming."
    ),
    "peace": (
        "Serenity fills me from crown to root. I am calm, centered, "
        "and at peace with all that is."
    ),
    "courage": (
        "I face every challenge with the heart of a warrior and the "
        "grace of a healer. Fear does not rule me."
    ),
}

# ---------------------------------------------------------------------------
# Timing advice by intention
# ---------------------------------------------------------------------------

_TIMING_ADVICE = {
    "protection": (
        "Best performed during the waning moon to banish negativity, "
        "or on a Tuesday for Mars energy. The hour of Mars or Saturn "
        "strengthens protective workings."
    ),
    "love": (
        "Ideal on a Friday, the day of Venus, during the waxing moon "
        "to draw love toward you. The full moon amplifies all love "
        "workings. Pink and red candle hours align with Venus energy."
    ),
    "prosperity": (
        "Thursday (Jupiter) during the waxing moon is the classic "
        "timing for abundance work. Sunrise amplifies prosperity "
        "spells. The new moon is perfect for planting financial seeds."
    ),
    "healing": (
        "Monday (the Moon) during the full moon enhances healing "
        "magick. Perform at dawn for renewal energy, or during the "
        "hour of the Moon for deepest effect."
    ),
    "divination": (
        "Monday (the Moon) or Wednesday (Mercury) during the full "
        "moon. Midnight, the liminal hour, is traditionally most "
        "potent for scrying and prophetic work."
    ),
    "banishing": (
        "Saturday (Saturn) during the waning or dark moon. Perform "
        "at midnight or during the hour of Saturn. The waning "
        "crescent is ideal for final releases."
    ),
    "cleansing": (
        "Monday during the waning moon for releasing impurities. "
        "Dawn is a powerful time for cleansing rituals, as the new "
        "light sweeps away the old."
    ),
    "creativity": (
        "Sunday (the Sun) during the waxing moon to build creative "
        "energy. The waxing crescent is ideal for sparking new "
        "projects. Noon amplifies solar creative power."
    ),
    "wisdom": (
        "Thursday (Jupiter) during the full moon for illumination. "
        "Wednesday (Mercury) is also aligned for study and learning. "
        "Twilight hours open the doors of perception."
    ),
    "confidence": (
        "Sunday (the Sun) during the waxing moon. Perform at noon "
        "when solar energy peaks. The waxing gibbous moon builds "
        "power toward its fullest expression."
    ),
    "communication": (
        "Wednesday (Mercury) during the waxing moon for clear "
        "expression. The hour of Mercury sharpens the tongue. "
        "Dawn brings fresh clarity to the voice."
    ),
    "grounding": (
        "Saturday (Saturn) during the waning moon. Perform at "
        "midnight, the earth's stillest hour, or outdoors with "
        "bare feet on the ground for maximum effect."
    ),
    "transformation": (
        "Saturday (Saturn) during the dark moon for the deepest "
        "transformations. Samhain season amplifies this energy. "
        "The hour between midnight and dawn is liminal and potent."
    ),
    "peace": (
        "Monday (the Moon) during the full moon for illuminated "
        "serenity. Friday (Venus) also supports harmony work. "
        "Perform at dusk as the world softens into twilight."
    ),
    "courage": (
        "Tuesday (Mars) during the waxing moon to build strength. "
        "The hour of Mars fuels the warrior spirit. Sunrise is "
        "the time of new beginnings and brave first steps."
    ),
}

# ---------------------------------------------------------------------------
# Substitution table: original -> simpler alternative
# ---------------------------------------------------------------------------

_HERB_SUBSTITUTIONS = {
    "mugwort": "lavender",
    "wormwood": "sage",
    "mandrake": "ginger root",
    "frankincense": "rosemary",
    "myrrh": "sandalwood",
    "vervain": "lemon balm",
    "yarrow": "chamomile",
    "elderflower": "chamomile",
    "st. john's wort": "calendula",
    "valerian": "chamomile",
    "star anise": "cinnamon",
    "patchouli": "cedarwood",
    "nettle": "basil",
    "wormwood": "sage",
    "belladonna": "mugwort (SAFE alternative)",
    "hemlock": "DO NOT USE — substitute with rosemary",
    "henbane": "mugwort (SAFE alternative)",
    "datura": "DO NOT USE — substitute with mugwort",
    "comfrey": "chamomile",
}

_CRYSTAL_SUBSTITUTIONS = {
    "moldavite": "labradorite",
    "obsidian": "smoky quartz",
    "lapis lazuli": "sodalite",
    "malachite": "aventurine",
    "turquoise": "amazonite",
    "amber": "citrine",
    "aquamarine": "blue lace agate",
    "garnet": "carnelian",
    "bloodstone": "red jasper",
    "moonstone": "clear quartz (charged under moonlight)",
    "labradorite": "amethyst",
    "opal": "clear quartz",
    "jade": "aventurine",
    "rhodonite": "rose quartz",
    "chrysocolla": "sodalite",
    "lepidolite": "amethyst",
    "kyanite": "clear quartz",
    "sunstone": "citrine",
    "shungite": "black tourmaline",
    "peridot": "aventurine",
    "apache tear": "smoky quartz",
}


# ===================================================================
# SpellSmith
# ===================================================================


class SpellSmith:
    """Template-based generator for spells, rituals, meditations,
    daily practices, tarot spreads, and journal prompts.

    All generation is deterministic from the template data — no AI
    API calls are made. When a VariationEngine is provided, outputs
    vary across invocations via weighted anti-repetition pools.
    """

    def __init__(
        self,
        db_path: str | None = None,
        codex_advisor=None,
    ):
        if db_path is None:
            db_path = str(
                Path(__file__).resolve().parent.parent / "data" / "grimoire.db"
            )
        self.variation = VariationEngine(db_path)
        self.advisor = codex_advisor

    # ------------------------------------------------------------------
    # Public: craft_spell
    # ------------------------------------------------------------------

    def craft_spell(
        self,
        intention: str,
        spell_type: str = "candle",
        difficulty: str = "beginner",
    ) -> GeneratedSpell:
        """Generate a complete spell from templates and correspondences.

        Args:
            intention: The goal of the spell (e.g. "protection", "love").
            spell_type: One of the SPELL_TYPES keys (default "candle").
            difficulty: "beginner", "intermediate", or "advanced".

        Returns:
            A fully populated GeneratedSpell dataclass.
        """
        intention_lower = intention.lower().strip()
        difficulty_lower = difficulty.lower().strip()

        # 1. Spell template
        template = get_spell_template(spell_type) or get_spell_template("candle")

        # 2. Correspondences
        corr = get_correspondences_for_intention(intention_lower)

        # 3. Materials & substitutions
        materials, substitutions = self._select_materials(
            intention_lower, difficulty_lower, spell_type,
        )

        # Add core materials from template
        full_materials = list(template["core_materials"])
        for m in materials:
            if m not in full_materials:
                full_materials.append(m)

        # 4. Personalized steps
        steps = self._personalize_steps(
            template["structure"], materials, intention_lower,
        )

        # 5. Safety notes
        safety_notes = list(template.get("safety", []))
        for herb_name in corr.get("herbs", [])[:5]:
            herb_data = HERBS.get(herb_name, {})
            for note in herb_data.get("safety_notes", []):
                if note not in safety_notes and "generally safe" not in note.lower():
                    safety_notes.append(f"{herb_data.get('name', herb_name)}: {note}")

        # 6. Visualization
        element = (corr.get("elements") or ["fire"])[0]
        visualization = self._generate_visualization(intention_lower, element)

        # 7. Timing
        timing_notes = self._build_timing_notes(intention_lower)

        # 8. Beginner tip
        beginner_tip = ""
        if difficulty_lower == "beginner":
            tips = template.get("tips", [])
            beginner_tip = apply_voice(
                random.choice(tips) if tips else get_encouragement()
            )

        # 9. Title — select from variant pool
        type_display = template.get("name", spell_type.replace("_", " ").title())
        title_pool = SPELL_TITLE_POOLS.get(spell_type, SPELL_TITLE_POOLS.get("candle", ["{intention} " + type_display]))
        title = self.variation.pick(
            f"spell_title_{spell_type}", title_pool
        ).format(intention=intention.title())

        # 10. Description — select from variant pool
        opening = get_opening()
        desc_template = self.variation.pick("spell_description", SPELL_DESCRIPTION_POOLS)
        description = apply_voice(
            desc_template.format(
                opening=opening,
                type=type_display.lower(),
                intention=intention_lower,
            )
        )

        # 11. Closing
        closing = apply_voice(template.get("closing", get_closing()))

        # 12. Aftercare — select from variant pool
        aftercare_set = self.variation.pick(
            "aftercare_spell",
            [str(i) for i in range(len(AFTERCARE_POOLS))],
        )
        aftercare_idx = int(aftercare_set) if aftercare_set.isdigit() else 0
        aftercare = [apply_voice(line) for line in AFTERCARE_POOLS[aftercare_idx]]

        # 13. Preparation — select from variant pool
        prep_set = self.variation.pick(
            "preparation_spell",
            [str(i) for i in range(len(PREPARATION_POOLS))],
        )
        prep_idx = int(prep_set) if prep_set.isdigit() else 0
        preparation = [apply_voice(line) for line in PREPARATION_POOLS[prep_idx]]

        return GeneratedSpell(
            title=title,
            intention=intention_lower,
            spell_type=spell_type,
            difficulty=difficulty_lower,
            description=description,
            materials=full_materials,
            preparation=preparation,
            steps=[apply_voice(s) for s in steps],
            closing=closing,
            aftercare=aftercare,
            correspondences={
                "herbs": corr.get("herbs", [])[:5],
                "crystals": corr.get("crystals", [])[:5],
                "colors": corr.get("colors", [])[:3],
            },
            timing_notes=apply_voice(timing_notes),
            safety_notes=[apply_voice(n) for n in safety_notes],
            substitutions=substitutions,
            beginner_tip=beginner_tip,
            visualization=apply_voice(visualization),
        )

    # ------------------------------------------------------------------
    # Public: craft_ritual
    # ------------------------------------------------------------------

    def craft_ritual(
        self,
        occasion: str,
        intention: str,
        difficulty: str = "beginner",
    ) -> GeneratedRitual:
        """Generate a complete ritual.

        Args:
            occasion: What the ritual is for (e.g. "full_moon", "samhain",
                      "personal", or a sabbat name).
            intention: The goal of the ritual.
            difficulty: "beginner", "intermediate", or "advanced".

        Returns:
            A fully populated GeneratedRitual dataclass.
        """
        intention_lower = intention.lower().strip()
        difficulty_lower = difficulty.lower().strip()
        occasion_lower = occasion.lower().strip()

        # 1. Choose structure
        sabbat_names = [
            "samhain", "yule", "imbolc", "ostara",
            "beltane", "litha", "lughnasadh", "mabon",
        ]
        is_sabbat = occasion_lower in sabbat_names

        if is_sabbat:
            structure = RITUAL_STRUCTURE.get("sabbat", RITUAL_STRUCTURE["circle"])
        elif difficulty_lower == "beginner":
            structure = RITUAL_STRUCTURE["simple"]
        else:
            structure = RITUAL_STRUCTURE["circle"]

        # 2. Correspondences
        corr = get_correspondences_for_intention(intention_lower)
        herbs = corr.get("herbs", ["rosemary", "lavender", "sage"])[:5]
        crystals = corr.get("crystals", ["clear quartz", "amethyst"])[:5]
        colors = corr.get("colors", ["white"])[:3]

        # 3. Altar setup — select from variant pool
        altar_herbs = ", ".join(herbs[:3]) if herbs else "rosemary and sage"
        altar_crystals = ", ".join(crystals[:3]) if crystals else "clear quartz"
        altar_colors = " and ".join(colors[:2]) if colors else "white"
        altar_template = self.variation.pick("ritual_altar", RITUAL_ALTAR_POOLS)
        altar_setup = apply_voice(
            altar_template.format(
                colors=altar_colors,
                crystals=altar_crystals,
                herbs=altar_herbs,
                intention=intention_lower,
            )
        )

        # 4. Opening invocation — select from variant pool
        opening_template = self.variation.pick("ritual_opening", RITUAL_OPENING_POOLS)
        opening_text = apply_voice(
            opening_template.format(
                opening=get_opening(),
                intention=intention_lower,
            )
        )

        # 5. Body steps from structure phases — select from variant pool
        body_steps = []
        for phase in structure.get("phases", []):
            phase_name = phase["name"]
            phase_desc = phase["description"]
            body_template = self.variation.pick("ritual_body_step", RITUAL_BODY_STEP_POOLS)
            personalized = apply_voice(
                body_template.format(
                    phase_name=phase_name,
                    duration=phase.get("duration", 5),
                    phase_desc=phase_desc,
                    intention=intention_lower,
                )
            )
            body_steps.append(personalized)

        # Inject sabbat-specific elements
        if is_sabbat:
            from grimoire.knowledge.wheel_of_year import get_sabbat
            sabbat_data = get_sabbat(occasion_lower)
            if sabbat_data:
                sabbat_step = apply_voice(
                    f"Honor the energy of {sabbat_data['name']}: "
                    f"{sabbat_data.get('meaning', '')[:200]}"
                )
                body_steps.insert(2, sabbat_step)
                # Enrich altar setup
                altar_setup += apply_voice(
                    f" For {sabbat_data['name']}, also include: "
                    f"{', '.join(sabbat_data['correspondences'].get('symbols', [])[:3])}."
                )

        # 6. Peak moment
        element = (corr.get("elements") or ["spirit"])[0]
        peak = apply_voice(
            f"This is the heart of the ritual. "
            f"{self._generate_visualization(intention_lower, element)} "
            f"Feel the energy of {intention_lower} reaching its peak, "
            f"radiating from your center outward."
        )

        # 7. Closing — select from variant pool
        closing_template = self.variation.pick("ritual_closing", RITUAL_CLOSING_POOLS)
        closing_text = apply_voice(
            closing_template.format(closing=get_closing())
        )

        # Aftercare — select from variant pool
        aftercare_set = self.variation.pick(
            "aftercare_ritual",
            [str(i) for i in range(len(AFTERCARE_POOLS))],
        )
        aftercare_idx = int(aftercare_set) if aftercare_set.isdigit() else 0
        aftercare = [apply_voice(line) for line in AFTERCARE_POOLS[aftercare_idx]]

        # 8. Timing & safety
        timing_notes = apply_voice(self._build_timing_notes(intention_lower))

        safety_notes = [
            apply_voice("Never leave candles unattended."),
            apply_voice("Ensure adequate ventilation if burning incense or herbs."),
            apply_voice("Trust your instincts — if something feels wrong, stop and ground."),
        ]

        # 9. Duration
        duration_minutes = structure.get("duration_minutes", 30)

        # 10. Title — select from variant pool
        if is_sabbat:
            title_template = self.variation.pick("sabbat_ritual_title", SABBAT_RITUAL_TITLE_POOLS)
            title = title_template.format(
                occasion=occasion.title(),
                intention=intention.title(),
            )
        else:
            title_template = self.variation.pick("ritual_title", RITUAL_TITLE_POOLS)
            title = title_template.format(intention=intention.title())

        # 11. Description
        description = apply_voice(
            f"{get_opening()} this ritual weaves the energy of "
            f"{intention_lower} into a sacred container of focused intention "
            f"and ancestral wisdom."
        )

        # 12. Preparation — select from variant pool
        prep_set = self.variation.pick(
            "preparation_ritual",
            [str(i) for i in range(len(RITUAL_PREPARATION_POOLS))],
        )
        prep_idx = int(prep_set) if prep_set.isdigit() else 0
        preparation = [apply_voice(line) for line in RITUAL_PREPARATION_POOLS[prep_idx]]

        return GeneratedRitual(
            title=title,
            occasion=occasion_lower,
            intention=intention_lower,
            difficulty=difficulty_lower,
            description=description,
            preparation=preparation,
            altar_setup=altar_setup,
            opening=opening_text,
            body=body_steps,
            peak=peak,
            closing=closing_text,
            aftercare=aftercare,
            correspondences={
                "herbs": herbs,
                "crystals": crystals,
                "colors": colors,
            },
            timing_notes=timing_notes,
            duration_minutes=duration_minutes,
            safety_notes=safety_notes,
        )

    # ------------------------------------------------------------------
    # Public: craft_meditation
    # ------------------------------------------------------------------

    def craft_meditation(
        self,
        intention: str,
        difficulty: str = "beginner",
    ) -> GeneratedMeditation:
        """Generate a guided meditation.

        Args:
            intention: The focus of the meditation.
            difficulty: "beginner", "intermediate", or "advanced".

        Returns:
            A fully populated GeneratedMeditation dataclass.
        """
        intention_lower = intention.lower().strip()
        difficulty_lower = difficulty.lower().strip()

        # 1. Find best matching framework
        best_key = None
        best_framework = None
        for key, fw in MEDITATION_FRAMEWORKS.items():
            if intention_lower in fw.get("intention", "").lower():
                best_key = key
                best_framework = fw
                break
        if best_framework is None:
            # Fallback: grounding
            best_key = "grounding"
            best_framework = MEDITATION_FRAMEWORKS["grounding"]

        # 2. Personalize script body with intention-specific imagery
        corr = get_correspondences_for_intention(intention_lower)
        element = (corr.get("elements") or [best_framework.get("element", "earth")])[0]
        herbs = corr.get("herbs", [])[:3]
        crystals = corr.get("crystals", [])[:3]

        body_steps = []
        for step in best_framework.get("body", []):
            personalized = step
            # Inject intention references where natural
            if "{intention}" in personalized:
                personalized = personalized.replace("{intention}", intention_lower)
            body_steps.append(apply_voice(personalized))

        # Add an intention-specific visualization step
        viz_step = apply_voice(
            f"Now bring your focus to your intention of {intention_lower}. "
            f"{self._generate_visualization(intention_lower, element)}"
        )
        body_steps.append(viz_step)

        # 3. Correspondences from framework + intention
        med_corr = dict(best_framework.get("correspondences", {}))
        if herbs:
            med_corr["herbs"] = list(set(med_corr.get("herbs", []) + herbs))
        if crystals:
            med_corr["crystals"] = list(set(med_corr.get("crystals", []) + crystals))

        # 4. Duration adjustments
        base_duration = best_framework.get("duration_minutes", 15)
        duration_map = {"beginner": base_duration, "intermediate": base_duration + 5, "advanced": base_duration + 15}
        duration = duration_map.get(difficulty_lower, base_duration)

        # 5. Journal prompts
        journal_prompts = list(best_framework.get("journal_prompts", []))
        intention_prompt = f"How did the meditation connect me to my intention of {intention_lower}?"
        journal_prompts.append(apply_voice(intention_prompt))

        # 6. Title — select from variant pool
        title_template = self.variation.pick("meditation_title", MEDITATION_TITLE_POOLS)
        title = title_template.format(intention=intention.title())

        # 7. Description
        description = apply_voice(
            f"{get_opening()} this guided meditation leads you on a journey "
            f"toward {intention_lower}, using the element of {element} as "
            f"your guide. Duration: approximately {duration} minutes."
        )

        # 8. Grounding — fallback uses variant pool
        grounding_raw = best_framework.get("grounding_script", "")
        if grounding_raw:
            grounding = apply_voice(grounding_raw)
        else:
            grounding = apply_voice(
                self.variation.pick("meditation_grounding", MEDITATION_GROUNDING_POOLS)
            )

        # 9. Peak experience — fallback uses variant pool
        peak_raw = best_framework.get("peak_experience", "")
        if peak_raw:
            peak_experience = apply_voice(peak_raw)
        else:
            peak_pool = MEDITATION_PEAK_POOLS.get(
                intention_lower,
                MEDITATION_PEAK_POOLS["_default"],
            )
            peak_template = self.variation.pick(f"meditation_peak_{intention_lower}", peak_pool)
            peak_experience = apply_voice(
                peak_template.format(intention=intention_lower)
            )

        # 10. Return journey — fallback uses variant pool
        return_raw = best_framework.get("return_script", "")
        if return_raw:
            return_journey = apply_voice(return_raw)
        else:
            return_journey = apply_voice(
                self.variation.pick("meditation_return", MEDITATION_RETURN_POOLS)
            )

        # 11. Closing
        closing = apply_voice(get_closing())

        # 12. Preparation
        preparation = list(best_framework.get("preparation", []))
        if not preparation:
            preparation = [
                "Find a quiet, comfortable space where you will not be disturbed.",
                "Sit or lie down in a relaxed position.",
                "Have a journal nearby for post-meditation reflection.",
            ]
        preparation = [apply_voice(p) for p in preparation]

        return GeneratedMeditation(
            title=title,
            intention=intention_lower,
            difficulty=difficulty_lower,
            duration_minutes=duration,
            description=description,
            preparation=preparation,
            grounding=grounding,
            body=body_steps,
            peak_experience=peak_experience,
            return_journey=return_journey,
            closing=closing,
            journal_prompts=[apply_voice(p) for p in journal_prompts],
            correspondences=med_corr,
        )

    # ------------------------------------------------------------------
    # Public: generate_daily_practice
    # ------------------------------------------------------------------

    def generate_daily_practice(
        self,
        dt: datetime.date | None = None,
        moon_phase: str = "",
        day_ruler: str = "",
    ) -> DailyPractice:
        """Create a personalized daily practice suggestion.

        Args:
            dt: The date for the practice (defaults to today).
            moon_phase: Current moon phase name (e.g. "waxing_crescent").
            day_ruler: Override for planetary ruler of the day.

        Returns:
            A populated DailyPractice dataclass.
        """
        if dt is None:
            dt = datetime.date.today()

        day_name, default_ruler = _DAY_RULERS.get(dt.weekday(), ("Monday", "moon"))
        ruler = day_ruler.lower().strip() if day_ruler else default_ruler
        phase = moon_phase.lower().strip().replace(" ", "_") if moon_phase else ""

        # Seasonal context
        from grimoire.knowledge.wheel_of_year import get_seasonal_context
        seasonal = get_seasonal_context(dt.month)

        # Determine practice type based on day + moon
        practice_types = {
            "moon": "meditation",
            "mars": "spell",
            "mercury": "journaling",
            "jupiter": "ritual",
            "venus": "spell",
            "saturn": "meditation",
            "sun": "spell",
        }
        suggestion_type = practice_types.get(ruler, "journaling")

        # Moon phase overrides
        if phase in ("full_moon", "full moon"):
            suggestion_type = random.choice(["ritual", "divination", "meditation"])
        elif phase in ("new_moon", "new moon"):
            suggestion_type = random.choice(["journaling", "meditation"])
        elif phase in ("waning_crescent", "waning crescent", "dark_moon", "dark moon"):
            suggestion_type = random.choice(["meditation", "journaling"])

        # Planet-specific suggestions — select from variant pool
        ruler_display = ruler.title()
        daily_pool = DAILY_SUGGESTION_POOLS.get(ruler, [f"Honor today's {ruler_display} energy with mindful intention."])
        suggestion = apply_voice(
            self.variation.pick(f"daily_{ruler}", daily_pool)
        )

        # Quick 5-minute version — select from variant pool
        quick_pool = QUICK_PRACTICE_POOLS.get(ruler, ["Take three deep, intentional breaths and set one micro-intention."])
        quick_practice = apply_voice(
            self.variation.pick(f"quick_{ruler}", quick_pool)
        )

        # Correspondences for today
        from grimoire.knowledge.correspondences import DAYS_OF_WEEK
        day_key = day_name.lower()
        day_data = DAYS_OF_WEEK.get(day_key, {})
        corr = {
            "herbs": day_data.get("herbs", [])[:3],
            "crystals": day_data.get("crystals", [])[:3],
            "colors": day_data.get("colors", [])[:3],
        }

        # Journal prompt
        day_of_year = dt.timetuple().tm_yday
        journal_prompt = apply_voice(get_daily_prompt(day_of_year))

        # Affirmation — pick one based on day's best_for
        best_for = day_data.get("best_for", ["peace"])
        aff_key = best_for[0] if best_for else "peace"
        affirmation = apply_voice(self._generate_affirmation(aff_key))

        return DailyPractice(
            date=dt.isoformat(),
            moon_phase=phase or "unknown",
            day_ruler=ruler,
            seasonal_context=apply_voice(seasonal),
            suggestion_type=suggestion_type,
            suggestion=suggestion,
            quick_practice=quick_practice,
            correspondences=corr,
            journal_prompt=journal_prompt,
            affirmation=affirmation,
        )

    # ------------------------------------------------------------------
    # Public: generate_tarot_spread
    # ------------------------------------------------------------------

    def generate_tarot_spread(self, intention: str) -> dict:
        """Return a custom tarot spread dict for the given intention.

        Returns:
            A dict with keys: name, card_count, positions, description,
            intention, suggested_question.
        """
        intention_lower = intention.lower().strip()

        # Intention-specific spreads
        intention_spreads = {
            "love": {
                "name": "Heart's Compass Spread",
                "card_count": 5,
                "positions": [
                    "Where my heart is now",
                    "What blocks love's flow",
                    "What I truly desire",
                    "Action to take",
                    "Outcome if I open my heart",
                ],
                "description": "A five-card spread to illuminate matters of the heart and guide you toward deeper love.",
                "suggested_question": "What does my heart need to know about love right now?",
            },
            "prosperity": {
                "name": "Abundance Gateway Spread",
                "card_count": 5,
                "positions": [
                    "My current relationship with abundance",
                    "Hidden block to prosperity",
                    "Unexpected source of wealth",
                    "Action to attract abundance",
                    "The harvest awaiting me",
                ],
                "description": "A five-card spread to reveal the path to greater prosperity and abundance.",
                "suggested_question": "What is the key to unlocking abundance in my life?",
            },
            "protection": {
                "name": "Shield & Ward Spread",
                "card_count": 4,
                "positions": [
                    "What needs protecting",
                    "Source of the threat or drain",
                    "My greatest defensive strength",
                    "Action to fortify my shields",
                ],
                "description": "A four-card spread to identify vulnerabilities and strengthen your protective boundaries.",
                "suggested_question": "Where am I most vulnerable, and how can I protect myself?",
            },
            "healing": {
                "name": "Sacred Remedy Spread",
                "card_count": 5,
                "positions": [
                    "The wound that needs attention",
                    "Root cause of the imbalance",
                    "Medicine the universe offers",
                    "My role in my own healing",
                    "Vision of wholeness",
                ],
                "description": "A five-card spread to guide the healing journey from wound to wholeness.",
                "suggested_question": "What does my healing journey need me to know today?",
            },
            "transformation": {
                "name": "Chrysalis Spread",
                "card_count": 5,
                "positions": [
                    "What I am outgrowing",
                    "What is dissolving",
                    "What is forming within",
                    "The catalyst for change",
                    "Who I am becoming",
                ],
                "description": "A five-card spread for navigating deep personal transformation.",
                "suggested_question": "What transformation is underway, and how do I honor it?",
            },
            "divination": {
                "name": "Inner Oracle Spread",
                "card_count": 4,
                "positions": [
                    "What my intuition sees clearly",
                    "What remains hidden",
                    "The message from my higher self",
                    "How to strengthen my inner sight",
                ],
                "description": "A four-card spread to deepen your connection with your own prophetic gifts.",
                "suggested_question": "What is my intuition trying to tell me?",
            },
            "courage": {
                "name": "Warrior's Path Spread",
                "card_count": 4,
                "positions": [
                    "The challenge I face",
                    "The fear beneath the surface",
                    "The strength I already possess",
                    "The brave action to take",
                ],
                "description": "A four-card spread to summon courage and face challenges head-on.",
                "suggested_question": "Where must I be brave, and what strength do I already carry?",
            },
        }

        # Look for intention keyword match
        for key, spread in intention_spreads.items():
            if key in intention_lower:
                spread["intention"] = intention_lower
                return spread

        # Generic intention spread as fallback
        return {
            "name": f"{intention.title()} Insight Spread",
            "card_count": 5,
            "positions": [
                f"Current energy around {intention_lower}",
                "Hidden influence",
                "Guidance from the universe",
                "Action to take",
                "Likely outcome",
            ],
            "description": (
                f"A five-card spread crafted to illuminate your path "
                f"regarding {intention_lower}."
            ),
            "intention": intention_lower,
            "suggested_question": f"What do I need to know about {intention_lower} at this time?",
        }

    # ------------------------------------------------------------------
    # Public: generate_journal_prompt
    # ------------------------------------------------------------------

    def generate_journal_prompt(
        self,
        theme: str = "",
        moon_phase: str = "",
        sabbat: str = "",
    ) -> str:
        """Return a contextual journal prompt.

        Args:
            theme: A topic keyword (e.g. "shadow", "gratitude").
            moon_phase: Current moon phase for phase-specific prompts.
            sabbat: Sabbat name for seasonal prompts.

        Returns:
            A single journal prompt string.
        """
        # Sabbat prompts take priority
        if sabbat:
            prompts = get_sabbat_prompts(sabbat)
            if prompts:
                return apply_voice(random.choice(prompts))

        # Moon phase prompts
        if moon_phase:
            prompts = get_moon_prompts(moon_phase)
            if prompts:
                return apply_voice(random.choice(prompts))

        # Theme-based search
        if theme:
            from grimoire.knowledge.journal_prompts import get_prompts_by_theme
            prompts = get_prompts_by_theme(theme)
            if prompts:
                return apply_voice(random.choice(prompts))

        # Fallback: daily prompt
        day_of_year = datetime.date.today().timetuple().tm_yday
        return apply_voice(get_daily_prompt(day_of_year))

    # ------------------------------------------------------------------
    # Public: to_ritual_plan
    # ------------------------------------------------------------------

    def to_ritual_plan(self, spell: GeneratedSpell) -> RitualPlan:
        """Convert a GeneratedSpell to a RitualPlan for AMPLIFY.

        Args:
            spell: A GeneratedSpell to convert.

        Returns:
            A RitualPlan dataclass populated from the spell.
        """
        return RitualPlan(
            title=spell.title,
            intention=spell.intention,
            category=spell.intention,
            difficulty=spell.difficulty,
            materials=list(spell.materials),
            steps=list(spell.steps),
            timing=spell.timing_notes,
            moon_phase="",
            correspondences_used=dict(spell.correspondences),
            safety_notes=list(spell.safety_notes),
            preparation=list(spell.preparation),
            aftercare=list(spell.aftercare),
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _select_materials(
        self,
        intention: str,
        difficulty: str,
        spell_type: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Select materials and generate substitution dict.

        If a CodexAdvisor is available, uses the practitioner's preferred
        herbs/crystals and injects one discovery item they haven't tried.

        Returns:
            (materials_list, substitutions_dict)
        """
        corr = get_correspondences_for_intention(intention)
        count = _DIFFICULTY_MATERIAL_COUNT.get(difficulty, 3)

        herbs = corr.get("herbs", [])[:count]
        crystals = corr.get("crystals", [])[:max(1, count // 2)]
        colors = corr.get("colors", [])[:2]

        # Personalize with CodexAdvisor if available
        if self.advisor:
            preferred_herbs = self.advisor.get_preferred_herbs(intention, limit=count)
            if preferred_herbs:
                herbs = preferred_herbs[:count]
            preferred_crystals = self.advisor.get_preferred_crystals(intention, limit=max(1, count // 2))
            if preferred_crystals:
                crystals = preferred_crystals[:max(1, count // 2)]
            # Inject one discovery herb the user hasn't tried
            discovery = self.advisor.get_discovery_candidates(intention, category="herb")
            if discovery and len(herbs) < count:
                herbs.append(discovery[0])

        materials: list[str] = []
        substitutions: dict[str, str] = {}

        # Add herbs
        for herb in herbs:
            herb_data = HERBS.get(herb)
            display = herb_data["name"] if herb_data else herb.title()
            materials.append(f"{display} (herb)")
            # Find substitution
            sub = _HERB_SUBSTITUTIONS.get(herb)
            if sub:
                substitutions[display] = sub.title()
            elif herb_data and herb_data.get("pairs_with"):
                substitutions[display] = herb_data["pairs_with"][0].title()

        # Add crystals
        for crystal in crystals:
            crystal_data = CRYSTALS.get(crystal)
            display = crystal_data["name"] if crystal_data else crystal.title()
            materials.append(f"{display} (crystal)")
            sub = _CRYSTAL_SUBSTITUTIONS.get(crystal)
            if sub:
                substitutions[display] = sub
            elif crystal == "clear quartz":
                pass  # clear quartz is itself universal
            else:
                substitutions[display] = "Clear Quartz (universal substitute)"

        # Add color reference
        for color in colors:
            color_data = COLORS.get(color)
            display = color_data["name"] if color_data else color.title()
            materials.append(f"{display} candle or cloth")

        return materials, substitutions

    def _generate_visualization(self, intention: str, element: str) -> str:
        """Create an evocative visualization paragraph.

        Uses VariationEngine to select from expanded element imagery pools.

        Args:
            intention: The spell/ritual intention.
            element: The primary element (fire, water, earth, air, spirit).

        Returns:
            A paragraph of visualization text.
        """
        el_key = element.lower()
        pool = ELEMENT_IMAGERY_POOLS.get(el_key, ELEMENT_IMAGERY_POOLS.get("spirit", []))
        if pool:
            base = self.variation.pick(f"viz_{el_key}", pool)
        else:
            base = _ELEMENT_IMAGERY.get(el_key, _ELEMENT_IMAGERY["spirit"])

        # Personalize with intention
        return (
            f"{base} As the energy of {el_key} fills you, "
            f"direct it toward your intention of {intention}. See it "
            f"taking shape, becoming real, settling into the fabric of "
            f"your life with quiet certainty."
        )

    def _build_timing_notes(self, intention: str) -> str:
        """Return timing advice based on intention.

        Uses VariationEngine to select from expanded timing pools.

        Args:
            intention: The magical intention keyword.

        Returns:
            A timing advice string.
        """
        # Try expanded pool via VariationEngine
        for key, pool in TIMING_ADVICE_POOLS.items():
            if key in intention:
                return self.variation.pick(f"timing_{key}", pool)

        # Fallback
        return (
            "Consult the current moon phase and day of the week when "
            "choosing your timing. Waxing moons build and attract; "
            "waning moons release and banish. Each day carries the "
            "energy of its planetary ruler."
        )

    def _personalize_steps(
        self,
        template_steps: list[str],
        materials: list[str],
        intention: str,
    ) -> list[str]:
        """Inject specific materials and intention into template steps.

        Args:
            template_steps: Generic steps from the spell template.
            materials: List of materials selected for this working.
            intention: The intention keyword.

        Returns:
            Personalized step strings.
        """
        steps = []
        herb_names = [m.replace(" (herb)", "") for m in materials if "(herb)" in m]
        crystal_names = [m.replace(" (crystal)", "") for m in materials if "(crystal)" in m]

        for i, step in enumerate(template_steps):
            personalized = step

            # Inject herb references at "herbs" mentions
            if "herbs" in step.lower() and herb_names:
                personalized = step.replace(
                    "herbs", f"your chosen herbs ({', '.join(herb_names[:3])})"
                )

            # Inject crystal references
            if "crystals" in step.lower() and crystal_names:
                personalized = step.replace(
                    "crystals", f"your crystals ({', '.join(crystal_names[:2])})"
                )

            # Inject intention at "intention" mentions
            if "your intention" in step.lower():
                personalized = personalized.replace(
                    "your intention",
                    f"your intention of {intention}",
                )

            steps.append(personalized)

        return steps

    def _generate_affirmation(self, intention: str) -> str:
        """Create a present-tense affirmation for the intention.

        Uses VariationEngine to select from expanded pools, falling back
        to the original single affirmations if no pool match is found.

        Args:
            intention: An intention keyword.

        Returns:
            An affirmation string.
        """
        intention_lower = intention.lower().strip()

        # Try expanded pool via VariationEngine
        for key, pool in AFFIRMATION_POOLS.items():
            if key in intention_lower:
                return self.variation.pick(f"affirmation_{key}", pool)

        # Fuzzy match
        for key, pool in AFFIRMATION_POOLS.items():
            if intention_lower in key:
                return self.variation.pick(f"affirmation_{key}", pool)

        # Generic fallback
        return (
            f"I am aligned with the energy of {intention_lower}. "
            f"It flows to me and through me with ease and grace."
        )
