"""
AMPLIFY Enhancement Pipeline for Rituals and Spells.

Six-stage pipeline that takes any RitualPlan and progressively enriches it,
making it comprehensive, safe, well-timed, and fully ready for practice.

Stages:
  1. ENRICH     - Inject correspondences, seasonal context, numerology
  2. EXPAND     - Add variants and adaptations for all skill levels
  3. FORTIFY    - Safety, substitutions, accessibility, ethical notes
  4. ANTICIPATE - Common challenges, preparation, aftercare
  5. OPTIMIZE   - Timing, energy amplifiers, alignment scoring
  6. VALIDATE   - Completeness check and readiness assessment

Adapted from VelvetVeilPrintables' AmplifyPDFPipeline for witchcraft
practice enhancement rather than PDF product generation.
"""

import time
from datetime import datetime

from grimoire.models import RitualPlan, AmplifyResult
from grimoire.knowledge.correspondences import (
    HERBS,
    CRYSTALS,
    COLORS,
    ELEMENTS,
    PLANETS,
    INTENTION_MAP,
    get_correspondences_for_intention,
)
from grimoire.knowledge.spell_templates import SPELL_TYPES
from grimoire.knowledge.moon_phases import MOON_PHASES, get_phase_data
from grimoire.knowledge.wheel_of_year import get_next_sabbat, get_seasonal_context
from grimoire.knowledge.numerology import get_magical_number
from grimoire.voice import get_closing, get_encouragement, VOICE_PROFILE


# ---------------------------------------------------------------------------
# Deity suggestions by intention
# ---------------------------------------------------------------------------

_DEITY_MAP = {
    "protection": [
        {"name": "Hecate", "tradition": "Greek", "description": "Goddess of crossroads, magick, and protection; guardian of thresholds"},
        {"name": "Thor", "tradition": "Norse", "description": "God of thunder; protector of humanity against chaos"},
        {"name": "Brigid", "tradition": "Celtic", "description": "Triple goddess of hearth, forge, and healing; protector of the home"},
    ],
    "love": [
        {"name": "Aphrodite", "tradition": "Greek", "description": "Goddess of love, beauty, and desire"},
        {"name": "Freya", "tradition": "Norse", "description": "Goddess of love, fertility, and seidr magick"},
        {"name": "Hathor", "tradition": "Egyptian", "description": "Goddess of love, joy, music, and motherhood"},
    ],
    "prosperity": [
        {"name": "Lakshmi", "tradition": "Hindu", "description": "Goddess of wealth, fortune, and abundance"},
        {"name": "Jupiter", "tradition": "Roman", "description": "King of the gods; ruler of expansion and fortune"},
        {"name": "Abundantia", "tradition": "Roman", "description": "Personification of abundance and prosperity"},
    ],
    "healing": [
        {"name": "Brigid", "tradition": "Celtic", "description": "Goddess of healing wells and sacred flame"},
        {"name": "Apollo", "tradition": "Greek", "description": "God of healing, light, and music"},
        {"name": "Isis", "tradition": "Egyptian", "description": "Great mother goddess; mistress of magick and healing"},
    ],
    "divination": [
        {"name": "Hecate", "tradition": "Greek", "description": "Goddess of magick, crossroads, and the liminal spaces"},
        {"name": "Odin", "tradition": "Norse", "description": "All-Father; sacrificed an eye for wisdom and mastery of the runes"},
        {"name": "Thoth", "tradition": "Egyptian", "description": "God of wisdom, writing, and magick; inventor of the tarot"},
    ],
    "banishing": [
        {"name": "Kali", "tradition": "Hindu", "description": "Goddess of destruction and transformation; destroyer of evil"},
        {"name": "The Morrigan", "tradition": "Celtic", "description": "Phantom queen; goddess of war, fate, and sovereignty"},
        {"name": "Hecate", "tradition": "Greek", "description": "Guardian of boundaries; she who keeps darkness at bay"},
    ],
    "cleansing": [
        {"name": "Brigid", "tradition": "Celtic", "description": "Purifier by sacred flame; keeper of the holy well"},
        {"name": "Saraswati", "tradition": "Hindu", "description": "Goddess of knowledge, purity, and flowing water"},
        {"name": "Selene", "tradition": "Greek", "description": "Moon goddess whose light cleanses and renews"},
    ],
    "creativity": [
        {"name": "Brigid", "tradition": "Celtic", "description": "Patroness of poetry, smithcraft, and creative inspiration"},
        {"name": "Apollo", "tradition": "Greek", "description": "God of music, poetry, and the arts"},
        {"name": "Saraswati", "tradition": "Hindu", "description": "Goddess of knowledge, music, and the creative arts"},
    ],
    "wisdom": [
        {"name": "Athena", "tradition": "Greek", "description": "Goddess of wisdom, strategy, and just warfare"},
        {"name": "Odin", "tradition": "Norse", "description": "Seeker of wisdom who hung on Yggdrasil for nine nights"},
        {"name": "Thoth", "tradition": "Egyptian", "description": "God of wisdom, writing, and keeper of cosmic order"},
    ],
    "confidence": [
        {"name": "Sekhmet", "tradition": "Egyptian", "description": "Lioness goddess of power, ferocity, and healing"},
        {"name": "Ares", "tradition": "Greek", "description": "God of courage, strength, and bold action"},
        {"name": "Freya", "tradition": "Norse", "description": "Goddess of war, love, and unyielding determination"},
    ],
    "communication": [
        {"name": "Hermes", "tradition": "Greek", "description": "Messenger god of eloquence, travel, and trickery"},
        {"name": "Thoth", "tradition": "Egyptian", "description": "God of language, writing, and sacred speech"},
        {"name": "Brigid", "tradition": "Celtic", "description": "Patroness of poets and inspired speech"},
    ],
    "grounding": [
        {"name": "Gaia", "tradition": "Greek", "description": "Primordial earth mother; the living ground beneath us"},
        {"name": "Cernunnos", "tradition": "Celtic", "description": "Horned god of the wild, forests, and the deep earth"},
        {"name": "Pan", "tradition": "Greek", "description": "God of nature, wilderness, and primal connection"},
    ],
    "transformation": [
        {"name": "Kali", "tradition": "Hindu", "description": "Goddess of time, change, and fierce transformation"},
        {"name": "Persephone", "tradition": "Greek", "description": "Queen of the underworld; she who descends and returns renewed"},
        {"name": "Cerridwen", "tradition": "Celtic", "description": "Keeper of the cauldron of transformation and rebirth"},
    ],
    "peace": [
        {"name": "Kuan Yin", "tradition": "Buddhist/Chinese", "description": "Bodhisattva of compassion and mercy"},
        {"name": "Pax", "tradition": "Roman", "description": "Goddess of peace and diplomatic harmony"},
        {"name": "Brigid", "tradition": "Celtic", "description": "Goddess of healing and the peaceful hearthfire"},
    ],
    "courage": [
        {"name": "Athena", "tradition": "Greek", "description": "Goddess of strategic courage and righteous strength"},
        {"name": "Thor", "tradition": "Norse", "description": "God of thunder; champion of bravery and protection"},
        {"name": "Sekhmet", "tradition": "Egyptian", "description": "Warrior lioness; embodiment of fearless power"},
    ],
}


# ---------------------------------------------------------------------------
# Optimal timing maps
# ---------------------------------------------------------------------------

_OPTIMAL_MOON_PHASE = {
    "protection": ("waning_crescent", "Waning energy is ideal for shielding and warding away what you do not want"),
    "love": ("waxing_gibbous", "Waxing energy draws love toward you as the moon builds to fullness"),
    "prosperity": ("waxing_gibbous", "Growing moonlight amplifies abundance and attraction of wealth"),
    "healing": ("full_moon", "Peak lunar energy provides maximum healing power and illumination"),
    "divination": ("full_moon", "The fully illuminated moon opens the veil and heightens psychic perception"),
    "banishing": ("waning_crescent", "As the moon diminishes, so does the power of what you banish"),
    "cleansing": ("last_quarter", "The releasing energy of the last quarter sweeps away impurities"),
    "creativity": ("waxing_crescent", "The young moon's emerging energy mirrors the birth of new ideas"),
    "wisdom": ("full_moon", "Full illumination reveals what was hidden and deepens understanding"),
    "confidence": ("waxing_gibbous", "Building lunar energy strengthens your inner fire and self-assurance"),
    "communication": ("first_quarter", "The decisive energy of the first quarter empowers clear expression"),
    "grounding": ("last_quarter", "Waning earth energy helps you settle deeply into stability"),
    "transformation": ("new_moon", "The darkness of the new moon is the womb of radical change"),
    "peace": ("full_moon", "The serene radiance of the full moon bestows calm and clarity"),
    "courage": ("first_quarter", "The half-lit moon challenges you to act boldly despite uncertainty"),
}

_OPTIMAL_DAY = {
    "protection": ("tuesday", "Mars", "Mars rules defense, courage, and protective fire"),
    "love": ("friday", "Venus", "Venus governs love, beauty, and the heart's desires"),
    "prosperity": ("thursday", "Jupiter", "Jupiter expands fortune, luck, and material abundance"),
    "healing": ("monday", "Moon", "The Moon nurtures, soothes, and restores"),
    "divination": ("monday", "Moon", "Lunar energy opens the psychic senses"),
    "banishing": ("saturday", "Saturn", "Saturn governs endings, boundaries, and banishment"),
    "cleansing": ("monday", "Moon", "Purification flows naturally under lunar influence"),
    "creativity": ("sunday", "Sun", "Solar energy ignites inspiration and creative fire"),
    "wisdom": ("thursday", "Jupiter", "Jupiter illuminates higher knowledge and truth"),
    "confidence": ("sunday", "Sun", "The Sun empowers the self and personal radiance"),
    "communication": ("wednesday", "Mercury", "Mercury rules eloquence, intellect, and exchange"),
    "grounding": ("saturday", "Saturn", "Saturn anchors energy into structure and stability"),
    "transformation": ("saturday", "Saturn", "Saturn governs the cycles of death and rebirth"),
    "peace": ("monday", "Moon", "The Moon's gentle energy calms and harmonizes"),
    "courage": ("tuesday", "Mars", "Mars emboldens the spirit and strengthens resolve"),
}


# ============================================================================
# AMPLIFY PIPELINE
# ============================================================================

class AmplifyPipeline:
    """Six-stage enhancement pipeline for RitualPlan objects.

    Usage:
        pipeline = AmplifyPipeline()
        plan = RitualPlan(
            title="Full Moon Protection Ritual",
            intention="protection",
            difficulty="beginner",
        )
        result = pipeline.amplify(plan)
        # result.ritual_plan now has enrichments, expansions, etc.
        # result.quality_score indicates overall readiness (0-100)

    Convenience:
        result = pipeline.amplify_quick("protection")
    """

    def __init__(self):
        pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def amplify(self, plan: RitualPlan) -> AmplifyResult:
        """Run all 6 AMPLIFY stages in sequence on *plan*.

        Populates ``plan.enrichments`` through ``plan.validations``, sets
        ``plan.amplified = True``, and returns an :class:`AmplifyResult`
        with ``quality_score``, ``processing_time_ms``, and readiness info.
        """
        start = time.time()
        result = AmplifyResult(ritual_plan=plan)

        try:
            # Stage 1: ENRICH
            plan.enrichments = self._enrich(plan)
            result.stages_completed.append("ENRICH")

            # Stage 2: EXPAND
            plan.expansions = self._expand(plan)
            result.stages_completed.append("EXPAND")

            # Stage 3: FORTIFY
            plan.fortifications = self._fortify(plan)
            result.stages_completed.append("FORTIFY")

            # Stage 4: ANTICIPATE
            plan.anticipations = self._anticipate(plan)
            result.stages_completed.append("ANTICIPATE")

            # Stage 5: OPTIMIZE
            plan.optimizations = self._optimize(plan)
            result.stages_completed.append("OPTIMIZE")

            # Stage 6: VALIDATE
            plan.validations = self._validate(plan)
            result.stages_completed.append("VALIDATE")

            plan.amplified = True
            result.ready = plan.validations.get("readiness_assessment", "").startswith("Ready")

        except Exception as exc:
            result.errors.append(f"Pipeline error at stage "
                                 f"{len(result.stages_completed) + 1}: {exc}")
            result.ready = False

        result.processing_time_ms = round((time.time() - start) * 1000, 2)
        result.quality_score = self._calculate_quality_score(result)
        return result

    def amplify_quick(self, intention: str, difficulty: str = "beginner") -> AmplifyResult:
        """Create a minimal RitualPlan from an intention and amplify it.

        This is a convenience entry point for callers who have only an
        intention string and an optional difficulty level.

        Args:
            intention: A short description such as ``"protection"`` or
                ``"love and healing"``.
            difficulty: One of ``"beginner"``, ``"intermediate"``, or
                ``"advanced"``.

        Returns:
            A fully amplified :class:`AmplifyResult`.
        """
        title = f"{intention.strip().title()} Ritual"
        plan = RitualPlan(
            title=title,
            intention=intention.strip().lower(),
            difficulty=difficulty,
            category=self._detect_primary_category(intention),
        )
        return self.amplify(plan)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 1 — ENRICH
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _enrich(self, plan: RitualPlan) -> dict:
        """Inject all relevant correspondences with reasons for each.

        Looks up the intention in INTENTION_MAP and the detailed knowledge
        modules, then assembles herb, crystal, color, element, planetary,
        deity, numerological, and seasonal context.
        """
        enrichments: dict = {}
        intention = plan.intention.lower()

        # --- Full correspondence set from INTENTION_MAP ---
        corr = get_correspondences_for_intention(intention)

        # --- Herbs with full details ---
        herb_details = []
        for herb_name in corr.get("herbs", [])[:8]:
            herb_data = HERBS.get(herb_name)
            if herb_data:
                herb_details.append({
                    "name": herb_data["name"],
                    "reason": f"Traditionally used for {', '.join(herb_data['magical_properties'][:3])}",
                    "element": herb_data.get("element", ""),
                    "planet": herb_data.get("planet", ""),
                    "safety_notes": herb_data.get("safety_notes", []),
                })
        enrichments["herbs"] = herb_details

        # --- Crystals with full details ---
        crystal_details = []
        for crystal_name in corr.get("crystals", [])[:6]:
            crystal_data = CRYSTALS.get(crystal_name)
            if crystal_data:
                crystal_details.append({
                    "name": crystal_data["name"],
                    "reason": f"Resonates with {', '.join(crystal_data['magical_properties'][:3])}",
                    "chakra": crystal_data.get("chakra", ""),
                    "cleansing_method": self._crystal_cleansing(crystal_name),
                })
        enrichments["crystals"] = crystal_details

        # --- Colors with candle use ---
        color_details = []
        for color_name in corr.get("colors", [])[:4]:
            color_data = COLORS.get(color_name)
            if color_data:
                color_details.append({
                    "name": color_data["name"],
                    "reason": f"Represents {', '.join(color_data['magical_properties'][:3])}",
                    "candle_use": color_data.get("candle_use", ""),
                })
        enrichments["colors"] = color_details

        # --- Elements ---
        element_details = []
        for el_name in corr.get("elements", [])[:2]:
            el_data = ELEMENTS.get(el_name)
            if el_data:
                element_details.append({
                    "name": el_data["name"],
                    "direction": el_data.get("direction", ""),
                    "tools": el_data.get("tools", []),
                    "qualities": el_data.get("qualities", []),
                })
        enrichments["elements"] = element_details

        # --- Planetary alignment ---
        planet_details = []
        for pl_name in corr.get("planets", [])[:2]:
            pl_data = PLANETS.get(pl_name)
            if pl_data:
                planet_details.append({
                    "name": pl_data["name"],
                    "day": pl_data.get("day", ""),
                    "metal": pl_data.get("metal", ""),
                    "magical_domains": pl_data.get("magical_domains", []),
                })
        enrichments["planetary_alignment"] = planet_details

        # --- Deity suggestions ---
        primary = self._detect_primary_category(intention)
        deities = _DEITY_MAP.get(primary, [])
        enrichments["deity_suggestions"] = deities

        # --- Numerological significance ---
        num_info = get_magical_number(intention)
        enrichments["numerology"] = {
            "number": num_info.get("number"),
            "meaning": num_info.get("meaning", ""),
            "magical_use": num_info.get("magical_use", ""),
            "tarot": num_info.get("tarot", ""),
            "recommended_repetitions": num_info.get("number"),
        }

        # --- Seasonal context ---
        now = datetime.now()
        seasonal_text = get_seasonal_context(now.month)
        next_name, next_data, days_until = get_next_sabbat(now.month, now.day)
        enrichments["seasonal_context"] = {
            "current_energy": seasonal_text,
            "next_sabbat": next_name,
            "days_until_sabbat": days_until,
            "sabbat_themes": next_data.get("themes", []) if next_data else [],
            "seasonal_tip": self._seasonal_tip(intention, now.month),
        }

        # --- Raw correspondence keys for downstream stages ---
        enrichments["_raw_correspondences"] = corr

        return enrichments

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 2 — EXPAND
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _expand(self, plan: RitualPlan) -> dict:
        """Add variants and adaptations for every skill level and context.

        Produces beginner, intermediate, advanced, quick, indoor, group,
        seasonal, and budget versions of the ritual.
        """
        intention = plan.intention.lower()
        expansions: dict = {}

        # --- Beginner version ---
        expansions["beginner_version"] = {
            "description": "Simplified 3-step version with minimal materials",
            "materials": self._beginner_materials(intention),
            "steps": [
                "Find a quiet space and take three deep breaths to center yourself.",
                f"Hold your chosen item (a simple candle or crystal) and state your "
                f"intention for {intention} clearly, aloud or in your mind.",
                "Sit quietly for five minutes, visualizing your intention as already "
                "fulfilled. When ready, take a grounding breath and close.",
            ],
            "duration_minutes": 10,
            "tip": "You do not need any tools to practice. Your intention is the most powerful ingredient.",
        }

        # --- Intermediate version ---
        herbs_used = [h["name"] for h in plan.enrichments.get("herbs", [])[:3]]
        crystals_used = [c["name"] for c in plan.enrichments.get("crystals", [])[:2]]
        colors_used = [c["name"] for c in plan.enrichments.get("colors", [])[:2]]
        expansions["intermediate_version"] = {
            "description": "Standard practice with full correspondences",
            "materials": (
                [f"{c} candle" for c in colors_used]
                + herbs_used
                + crystals_used
                + ["matches or lighter", "fireproof holder", "paper and pen"]
            ),
            "steps": [
                "Cleanse your space with smoke, sound, or salt water.",
                "Arrange your materials on your working surface.",
                f"Light your {colors_used[0].lower() if colors_used else 'white'} candle.",
                f"Hold your crystal(s) and speak your intention for {intention}.",
                "Place herbs around the candle base or in a small bowl.",
                "Meditate on your intention for 10-15 minutes, visualizing fully.",
                "Write your intention on paper and place it under the candle.",
                "Allow the candle to burn safely. Close with gratitude.",
            ],
            "duration_minutes": 30,
            "tip": "Match your candle color and herbs to your intention for maximum alignment.",
        }

        # --- Advanced version ---
        expansions["advanced_version"] = {
            "description": "Extended practice with deeper visualization, chanting, and multiple tools",
            "materials": (
                expansions["intermediate_version"]["materials"]
                + ["incense or resin on charcoal", "anointing oil", "compass or directional markers"]
            ),
            "steps": [
                "Ritually bathe or wash your hands with intention-infused water.",
                "Cast a circle or define your sacred space in your tradition.",
                "Call the quarters/elements, beginning in the East.",
                "Invoke any deities or spirit allies relevant to your working.",
                f"State your intention for {intention} in formal ritual language.",
                "Anoint your candle with oil, carve a sigil into it, and dress it with herbs.",
                "Light the candle and chant your intention, building energy with each repetition.",
                "At the peak of energy, release it into the universe with a sharp exhale or clap.",
                "Ground by placing your hands on the earth or eating something.",
                "Thank all beings invoked, release the quarters, and open the circle.",
                "Record your experience in your grimoire.",
            ],
            "duration_minutes": 60,
            "tip": "Build energy slowly through chanting or drumming before releasing at the peak.",
        }

        # --- Quick version ---
        expansions["quick_version"] = {
            "description": "5-minute version for busy days",
            "materials": ["Just yourself"],
            "steps": [
                "Close your eyes and take three deep breaths.",
                f"Place your hand on your heart and whisper: 'I call {intention} to me now.'",
                "Visualize a light in the color associated with your intention filling your body.",
                "Hold for one minute, breathing deeply.",
                "Open your eyes and carry the intention with you.",
            ],
            "duration_minutes": 5,
            "tip": "This version is perfect for a morning practice or before a stressful event.",
        }

        # --- Indoor adaptation ---
        expansions["indoor_adaptation"] = {
            "description": "How to practice this without outdoor space",
            "adjustments": [
                "Use a windowsill altar facing the appropriate direction.",
                "Replace outdoor offerings with a small bowl of water or salt.",
                "Use electric or battery-operated candles if open flame is not allowed.",
                "Play nature sounds or recordings of wind, rain, or birdsong.",
                "Use potted herbs instead of gathered ones.",
                "Open a window to invite fresh air and natural energy.",
            ],
        }

        # --- Group adaptation ---
        expansions["group_adaptation"] = {
            "description": "How to practice with 2 or more people",
            "adjustments": [
                "Assign elemental roles: one person per direction/element.",
                "Hold hands in a circle during the energy-raising phase.",
                "Take turns stating individual intentions within the shared working.",
                "Designate one person to lead the opening and another the closing.",
                "Share a simple meal or drink afterward to ground together.",
                "Each participant can bring one material to contribute to the collective altar.",
            ],
        }

        # --- Seasonal adaptation ---
        now = datetime.now()
        seasonal_energy = get_seasonal_context(now.month)
        expansions["seasonal_adaptation"] = {
            "description": f"Adjustments for the current season ({self._current_season(now.month)})",
            "current_seasonal_energy": seasonal_energy,
            "adjustments": self._seasonal_adjustments(intention, now.month),
        }

        # --- Budget version ---
        expansions["budget_version"] = {
            "description": "Free or very cheap alternatives for all materials",
            "substitutions": {
                "candle": "A birthday candle, a tea light, or even an LED light",
                "crystal": "A smooth river stone, a piece of glass, or visualization of the crystal",
                "herbs": "Kitchen spices (rosemary, basil, cinnamon, sage) or a simple cup of tea",
                "incense": "A bay leaf burned carefully on a plate, or boiling spices in water",
                "altar cloth": "A clean pillowcase, scarf, or napkin in the appropriate color",
                "anointing oil": "Olive oil from the kitchen, infused with intention",
                "special tools": "Your hands, your breath, and your voice are the only tools required",
                "offering bowl": "Any clean cup, mug, or jar",
            },
            "note": "Magick has been practiced by people of all economic backgrounds for millennia. "
                    "Intention and focus are more powerful than any purchased tool.",
        }

        return expansions

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 3 — FORTIFY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fortify(self, plan: RitualPlan) -> dict:
        """Safety, substitutions, accessibility, and ethics.

        Adds fire safety, herb/crystal warnings, material substitutions,
        accessibility adaptations, ethical notes, and a medical disclaimer.
        """
        fortifications: dict = {}
        intention = plan.intention.lower()

        # --- Fire safety ---
        uses_fire = self._involves_fire(plan)
        fortifications["fire_safety"] = {
            "applicable": uses_fire,
            "notes": [
                "Never leave a burning candle unattended.",
                "Use a sturdy, fireproof holder on a heat-resistant surface.",
                "Keep candles away from curtains, paper, and flammable materials.",
                "Snuff candles rather than blowing them out (preserves the magick and prevents sparks).",
                "Have water or a fire extinguisher accessible.",
                "If burning herbs or paper, use a fireproof bowl or cauldron.",
                "Trim candle wicks to 1/4 inch to prevent excessive flame.",
                "Consider LED candles as a safe alternative — intention matters more than flame.",
            ] if uses_fire else [
                "This working does not involve open flame. No fire safety precautions required.",
            ],
        }

        # --- Herb safety ---
        herb_safety = []
        for herb_entry in plan.enrichments.get("herbs", []):
            herb_name_lower = herb_entry.get("name", "").lower()
            herb_data = HERBS.get(herb_name_lower)
            if herb_data:
                warnings = herb_data.get("safety_notes", [])
                if warnings and warnings != ["Generally safe"]:
                    herb_safety.append({
                        "herb": herb_entry["name"],
                        "warnings": warnings,
                        "pregnancy_caution": any("pregnan" in w.lower() for w in warnings),
                        "pet_toxicity": any("toxic" in w.lower() or "pet" in w.lower() for w in warnings),
                        "allergy_risk": any("allerg" in w.lower() for w in warnings),
                    })
                else:
                    herb_safety.append({
                        "herb": herb_entry["name"],
                        "warnings": ["Generally considered safe for external and aromatic use."],
                        "pregnancy_caution": False,
                        "pet_toxicity": False,
                        "allergy_risk": False,
                    })
        fortifications["herb_safety"] = herb_safety

        # --- Crystal safety ---
        crystal_safety = []
        water_unsafe = {"selenite", "malachite", "pyrite", "lepidolite", "hematite",
                        "lapis lazuli", "fluorite", "turquoise", "chrysocolla", "kyanite",
                        "shungite", "amber"}
        toxic_crystals = {"malachite", "chrysocolla", "lepidolite", "pyrite",
                          "tiger's eye", "lapis lazuli"}
        for crystal_entry in plan.enrichments.get("crystals", []):
            crystal_name = crystal_entry.get("name", "")
            crystal_key = crystal_name.lower()
            is_water_unsafe = crystal_key in water_unsafe
            is_toxic = crystal_key in toxic_crystals
            crystal_data = CRYSTALS.get(crystal_key)
            notes = crystal_data.get("safety_notes", []) if crystal_data else []
            crystal_safety.append({
                "crystal": crystal_name,
                "water_safe": not is_water_unsafe,
                "water_warning": f"Do NOT place {crystal_name} in water; it may dissolve, rust, or release toxic minerals." if is_water_unsafe else "",
                "toxicity_warning": f"{crystal_name} contains minerals that can be harmful if ingested. Do not use in gem elixirs." if is_toxic else "",
                "handling_notes": notes,
            })
        fortifications["crystal_safety"] = crystal_safety

        # --- Substitutions ---
        substitutions = {}
        for herb_entry in plan.enrichments.get("herbs", []):
            herb_key = herb_entry.get("name", "").lower()
            herb_data = HERBS.get(herb_key)
            if herb_data:
                pairs = herb_data.get("pairs_with", [])
                alt_magical = pairs[0] if len(pairs) > 0 else "rosemary (universal substitute)"
                substitutions[herb_entry["name"]] = [
                    alt_magical.title(),
                    self._household_substitute_herb(herb_key),
                ]
        for crystal_entry in plan.enrichments.get("crystals", []):
            crystal_key = crystal_entry.get("name", "").lower()
            crystal_data = CRYSTALS.get(crystal_key)
            if crystal_data:
                pairs = crystal_data.get("pairs_with", [])
                alt_crystal = pairs[0].title() if len(pairs) > 0 else "Clear Quartz (universal substitute)"
                substitutions[crystal_entry["name"]] = [
                    alt_crystal,
                    "A smooth river stone or any stone that feels right in your hand",
                ]
        fortifications["substitutions"] = substitutions

        # --- Accessibility ---
        fortifications["accessibility"] = {
            "mobility_limitations": [
                "All rituals can be performed seated or lying down.",
                "Use a tray or lap desk as a portable altar.",
                "Guided visualization replaces any physical movement.",
                "Voice-activated devices can play chants or ambient music.",
                "Pre-arrange all materials within arm's reach before beginning.",
            ],
            "sensory_sensitivity": [
                "Replace incense or smoke with essential oil diffusers or sprays.",
                "Use unscented candles if fragrance is overwhelming.",
                "Dim lighting or use colored cloth over a lamp instead of candles.",
                "Written affirmations can replace spoken incantations.",
                "Noise-cancelling headphones with nature sounds for focus.",
            ],
            "apartment_living": [
                "Use LED candles or battery-powered tea lights.",
                "Simmer pots (water + herbs on the stove) replace smoke cleansing.",
                "Sound cleansing with a bell, singing bowl, or clapping works in any space.",
                "Windowsill altars and tabletop setups need minimal room.",
                "Salt lines at thresholds provide protection without altering the space.",
            ],
        }

        # --- Ethical notes ---
        fortifications["ethical_notes"] = [
            "'An it harm none, do what ye will.' Consider the ripple effects of every working.",
            "Love magick must never override another person's free will or consent. "
            "Focus on self-love, attracting compatible energy, or strengthening existing mutual bonds.",
            "Source herbs and crystals ethically. White sage is over-harvested from Indigenous lands; "
            "garden sage, rosemary, or cedar are excellent alternatives.",
            "Cultural respect: research the origins of practices you adopt. Credit traditions "
            "and avoid claiming sacred practices from closed or marginalized communities as your own.",
            "Never use magick to manipulate, control, or harm another being.",
            "Respect the land and environment when gathering materials outdoors. "
            "Take only what you need and leave an offering of gratitude.",
        ]

        # --- Medical disclaimer ---
        fortifications["medical_disclaimer"] = (
            "Magickal practice is a spiritual and personal discipline. It is not a "
            "substitute for professional medical, psychological, or legal advice. "
            "If you are experiencing a health crisis, mental health emergency, or "
            "legal situation, please seek qualified professional help. Herbs discussed "
            "here are for aromatic, ritual, and symbolic use unless otherwise noted; "
            "consult a qualified herbalist or physician before internal use."
        )

        return fortifications

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 4 — ANTICIPATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _anticipate(self, plan: RitualPlan) -> dict:
        """Common challenges, preparation, expectations, and aftercare.

        Predicts what might go wrong and prepares the practitioner with
        solutions, checklists, and self-care guidance.
        """
        anticipations: dict = {}
        intention = plan.intention.lower()

        # --- Common challenges ---
        anticipations["common_challenges"] = [
            {
                "challenge": "Candle won't stay lit",
                "solution": "Try a draft-free location. Cup your hands around the flame gently. "
                            "An LED candle is an equally valid substitute — the intention matters, not the flame.",
            },
            {
                "challenge": "Mind keeps wandering during meditation or visualization",
                "solution": "This is completely normal, especially for beginners. Gently return "
                            "your focus without judgment each time. Wandering is not failure; "
                            "noticing it is awareness. Try focusing on your breath as an anchor.",
            },
            {
                "challenge": "Don't feel anything during the working",
                "solution": "Magick does not require dramatic sensations. Many experienced "
                            "practitioners feel nothing in the moment. Trust the process and "
                            "look for subtle shifts in the days that follow.",
            },
            {
                "challenge": "Interrupted mid-ritual",
                "solution": "Pause, handle the interruption calmly, then return. You can "
                            "resume where you left off or close gracefully. The energy is "
                            "not lost — it simply waits for your attention.",
            },
            {
                "challenge": "Materials unavailable",
                "solution": "Intention is more powerful than any tool. Use substitutions from "
                            "the fortifications section, or practice with visualization alone. "
                            "A working done with full focus and no tools outshines a distracted "
                            "ritual with every correspondence perfectly matched.",
            },
            {
                "challenge": "Feeling anxious or emotional during the working",
                "solution": "This can happen, especially with transformation, healing, or "
                            "shadow work. Pause and ground yourself: feel your feet on the floor, "
                            "take slow breaths, hold a grounding stone. You can always stop and "
                            "return to the working another day.",
            },
            {
                "challenge": "Unsure if the words are 'right'",
                "solution": "There is no single correct incantation. Speak from your heart in "
                            "your own words. Authenticity is more powerful than perfection. "
                            "If you prefer structure, use the provided scripts as a starting point.",
            },
        ]

        # --- Preparation checklist ---
        anticipations["preparation_checklist"] = [
            "Choose your date and time (consult the timing guidance in optimizations).",
            "Gather all materials and place them in your working area.",
            "Cleanse your space: open a window, burn cleansing herbs, or sprinkle salt water.",
            "Cleanse yourself: wash your hands, take a ritual bath, or simply take three conscious breaths.",
            "Silence your phone and minimize potential interruptions.",
            "Set up your altar or working surface with materials arranged intentionally.",
            "Review the steps of your working so you are not reading mid-ritual.",
            "Ground and center: stand or sit quietly, feel your connection to the earth.",
            "State your intention clearly to yourself before beginning.",
        ]

        # --- What to expect ---
        anticipations["what_to_expect"] = {
            "during": [
                "A sense of calm focus or heightened awareness.",
                "Tingling in the hands, warmth, or subtle shifts in the air.",
                "Emotional release — tears, laughter, or a deep sigh are all normal.",
                "Nothing perceptible at all — this is equally valid and common.",
                "A feeling of 'rightness' or completion when the working is done.",
            ],
            "after": [
                "You may feel energized, calm, or pleasantly tired.",
                "Dreams may be more vivid in the nights following.",
                "Synchronicities related to your intention may appear in daily life.",
                "Changes often manifest subtly over days or weeks, not instantly.",
                "You may notice shifts in your own behavior, perspective, or confidence.",
            ],
        }

        # --- Aftercare ---
        anticipations["aftercare"] = [
            "Ground yourself: eat something, drink water, place your hands on the earth.",
            "Record your experience in a journal or grimoire while it is fresh.",
            "Rest if you feel drained — energy work can be tiring.",
            "Avoid immediately scrolling social media or engaging in stressful tasks.",
            "Take a warm shower or bath to wash away any residual energy.",
            "Spend a few minutes in gratitude for the practice itself.",
            "Be gentle with yourself for the rest of the day.",
        ]

        # --- Signs of success ---
        anticipations["signs_of_success"] = self._signs_of_success(intention)

        return anticipations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 5 — OPTIMIZE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _optimize(self, plan: RitualPlan) -> dict:
        """Timing, energy amplifiers, and alignment scoring.

        Recommends optimal moon phase, day of week, power boosters, and
        calculates how well the current plan aligns with ideal conditions.
        """
        optimizations: dict = {}
        intention = plan.intention.lower()
        primary = self._detect_primary_category(intention)

        # --- Optimal moon phase ---
        phase_key, phase_reason = _OPTIMAL_MOON_PHASE.get(
            primary, ("full_moon", "The full moon is universally powerful for all workings")
        )
        phase_data = get_phase_data(phase_key)
        optimizations["optimal_moon_phase"] = {
            "phase": phase_data["name"] if phase_data else phase_key.replace("_", " ").title(),
            "phase_key": phase_key,
            "reason": phase_reason,
            "best_for": phase_data.get("best_for", []) if phase_data else [],
        }

        # --- Optimal day ---
        day_name, planet_name, day_reason = _OPTIMAL_DAY.get(
            primary, ("thursday", "Jupiter", "Jupiter is expansive and supports many intentions")
        )
        optimizations["optimal_day"] = {
            "day": day_name.title(),
            "planet": planet_name,
            "reason": day_reason,
        }

        # --- Energy amplifiers ---
        optimizations["energy_amplifiers"] = [
            {
                "amplifier": "Location",
                "description": "Practice outdoors in nature, near water, or in a garden if possible. "
                               "Liminal spaces (doorways, crossroads, forest edges) amplify magick.",
            },
            {
                "amplifier": "Time of day",
                "description": self._best_time_of_day(primary),
            },
            {
                "amplifier": "Repetition",
                "description": f"Repeat this working for {plan.enrichments.get('numerology', {}).get('recommended_repetitions', 3)} "
                               f"consecutive days or at each corresponding moon phase for a full lunar cycle.",
            },
            {
                "amplifier": "Group energy",
                "description": "Working with others who share your intention multiplies the energy. "
                               "Even two people create significantly more power than one.",
            },
            {
                "amplifier": "Emotional intensity",
                "description": "The stronger your emotional connection to the intention, the more "
                               "potent the working. Choose a time when you feel the desire most acutely.",
            },
        ]

        # --- Power boosters ---
        optimizations["power_boosters"] = [
            {
                "booster": "Chanting or mantra",
                "description": "Repeat a short phrase that encapsulates your intention. Build speed "
                               "and volume as energy rises, then release into silence.",
            },
            {
                "booster": "Drumming",
                "description": "Steady rhythmic drumming raises energy and induces a light trance state. "
                               "Any drum, tambourine, or even tapping on a table works.",
            },
            {
                "booster": "Dance or movement",
                "description": "Move your body freely in a circle or in whatever way feels right. "
                               "Physical movement circulates and amplifies energy.",
            },
            {
                "booster": "Breathwork",
                "description": "Rhythmic deep breathing (such as 4-count inhale, 4-count hold, 4-count exhale) "
                               "centers the mind and charges the working with life force.",
            },
            {
                "booster": "Visualization",
                "description": "Create a vivid mental image of your intention fulfilled. Engage all senses: "
                               "see, hear, feel, smell, and taste the outcome as though it were real now.",
            },
        ]

        # --- Alignment score ---
        optimizations["alignment_score"] = self._calculate_alignment(plan, primary)

        # --- Seasonal boost ---
        now = datetime.now()
        optimizations["seasonal_boost"] = {
            "current_season": self._current_season(now.month),
            "seasonal_energy": get_seasonal_context(now.month),
            "alignment_with_intention": self._seasonal_alignment(intention, now.month),
        }

        return optimizations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 6 — VALIDATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _validate(self, plan: RitualPlan) -> dict:
        """Completeness check, readiness assessment, and encouragement.

        Runs an 8-point checklist and returns a final readiness verdict.
        """
        validations: dict = {}

        # --- 8-point checklist ---
        checklist = {}

        # 1. Clear intention
        has_intention = bool(plan.intention and len(plan.intention.strip()) > 0)
        checklist["clear_intention_stated"] = has_intention

        # 2. Materials listed with alternatives
        has_materials = bool(plan.enrichments.get("herbs") or plan.enrichments.get("crystals"))
        has_alternatives = bool(plan.fortifications.get("substitutions"))
        checklist["materials_with_alternatives"] = has_materials and has_alternatives

        # 3. Step-by-step instructions
        has_steps = bool(plan.steps) or bool(plan.expansions.get("intermediate_version", {}).get("steps"))
        checklist["step_by_step_instructions"] = has_steps

        # 4. Opening and closing
        has_opening_closing = bool(plan.expansions.get("advanced_version", {}).get("steps"))
        checklist["opening_and_closing"] = has_opening_closing

        # 5. Safety notes
        has_safety = bool(plan.fortifications.get("fire_safety") or plan.fortifications.get("herb_safety"))
        checklist["safety_notes_present"] = has_safety

        # 6. Timing guidance
        has_timing = bool(plan.optimizations.get("optimal_moon_phase") or plan.optimizations.get("optimal_day"))
        checklist["timing_guidance_included"] = has_timing

        # 7. Aftercare/grounding
        has_aftercare = bool(plan.anticipations.get("aftercare"))
        checklist["aftercare_described"] = has_aftercare

        # 8. Personalization options
        has_personalization = bool(
            plan.expansions.get("beginner_version")
            and plan.expansions.get("advanced_version")
            and plan.expansions.get("budget_version")
        )
        checklist["personalization_options"] = has_personalization

        validations["checklist"] = checklist

        # --- Intention alignment ---
        passed = sum(1 for v in checklist.values() if v)
        total = len(checklist)
        intention_alignment = round((passed / total) * 100, 1) if total > 0 else 0.0
        validations["intention_alignment"] = intention_alignment

        # --- Readiness assessment ---
        missing = [k.replace("_", " ").title() for k, v in checklist.items() if not v]
        if not missing:
            validations["readiness_assessment"] = "Ready to practice"
        else:
            validations["readiness_assessment"] = f"Needs attention: {', '.join(missing)}"

        # --- Encouragement ---
        validations["encouragement"] = get_encouragement()

        # --- Closing message ---
        validations["closing_message"] = get_closing()

        return validations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # QUALITY SCORE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calculate_quality_score(self, result: AmplifyResult) -> float:
        """Compute a 0-100 quality score based on pipeline completeness.

        Scoring breakdown:
          +10 per stage completed (max 60)
          +5 for correspondences enriched
          +3 for seasonal context
          +5 for 3+ variants
          +3 for budget version
          +5 for substitutions
          +3 for accessibility
          +3 for preparation checklist
          +2 for aftercare
          +3 for timing optimization
          +2 for power boosters
          +5 for passing all 8 validation checks
        Total theoretical max: ~99-100
        """
        score = 0.0
        plan = result.ritual_plan

        # Stages completed: +10 each, max 60
        score += len(result.stages_completed) * 10

        # Enrichments
        enrichments = plan.enrichments
        if enrichments.get("herbs") or enrichments.get("crystals") or enrichments.get("colors"):
            score += 5
        if enrichments.get("seasonal_context"):
            score += 3

        # Expansions
        expansions = plan.expansions
        variant_count = sum(1 for k in ("beginner_version", "intermediate_version",
                                         "advanced_version", "quick_version")
                           if expansions.get(k))
        if variant_count >= 3:
            score += 5
        if expansions.get("budget_version"):
            score += 3

        # Fortifications
        fortifications = plan.fortifications
        if fortifications.get("substitutions"):
            score += 5
        if fortifications.get("accessibility"):
            score += 3

        # Anticipations
        anticipations = plan.anticipations
        if anticipations.get("preparation_checklist"):
            score += 3
        if anticipations.get("aftercare"):
            score += 2

        # Optimizations
        optimizations = plan.optimizations
        if optimizations.get("optimal_moon_phase") or optimizations.get("optimal_day"):
            score += 3
        if optimizations.get("power_boosters"):
            score += 2

        # Validations
        validations = plan.validations
        checklist = validations.get("checklist", {})
        if checklist and all(checklist.values()):
            score += 5

        return min(score, 100.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _detect_primary_category(intention: str) -> str:
        """Extract the primary intention category from free text."""
        intention_lower = intention.lower()
        for keyword in INTENTION_MAP:
            if keyword in intention_lower:
                return keyword
        return "protection"  # safe default

    @staticmethod
    def _crystal_cleansing(crystal_key: str) -> str:
        """Return the recommended cleansing method for a crystal."""
        water_unsafe = {"selenite", "malachite", "pyrite", "lepidolite", "hematite",
                        "lapis lazuli", "fluorite", "turquoise", "chrysocolla",
                        "kyanite", "shungite"}
        if crystal_key in water_unsafe:
            return "Smoke cleansing, selenite plate, or moonlight only (do NOT use water)"
        return "Running water, moonlight, smoke, salt, or selenite plate"

    @staticmethod
    def _beginner_materials(intention: str) -> list[str]:
        """Return minimal materials for a beginner version."""
        base = ["A white candle (or LED candle)", "A quiet space"]
        intention_lower = intention.lower()
        if "protection" in intention_lower:
            base.append("A pinch of salt")
        elif "love" in intention_lower:
            base.append("A small piece of rose quartz or a pink item")
        elif "prosperity" in intention_lower:
            base.append("A coin or a pinch of cinnamon")
        elif "healing" in intention_lower:
            base.append("A glass of water")
        elif "divination" in intention_lower:
            base.append("A reflective surface (mirror, bowl of water)")
        else:
            base.append("A small object that represents your intention")
        return base

    @staticmethod
    def _household_substitute_herb(herb_key: str) -> str:
        """Return a common household substitute for an herb."""
        substitutes = {
            "sage": "dried kitchen sage from the spice rack",
            "rosemary": "fresh or dried rosemary from the kitchen",
            "basil": "dried basil from the spice rack",
            "cinnamon": "a cinnamon stick or ground cinnamon from the kitchen",
            "lavender": "lavender tea bag or a few drops of lavender essential oil",
            "chamomile": "chamomile tea bag",
            "bay laurel": "bay leaves from the spice rack",
            "mint": "fresh mint leaves or mint tea bag",
            "thyme": "dried thyme from the spice rack",
            "ginger": "fresh ginger root or ground ginger from the kitchen",
            "rose": "dried rose petals from a tea blend or a rose from a bouquet",
            "cedar": "a small piece of cedar wood or cedar chips (pet bedding section)",
            "mugwort": "chamomile or lavender as a gentler alternative",
            "frankincense": "a bay leaf burned on a plate",
            "myrrh": "frankincense resin or pine resin as a substitute",
            "sandalwood": "cedar or a sandalwood-scented item",
            "patchouli": "cedar chips or a pinch of dried garden soil",
            "jasmine": "jasmine tea bag or a few drops of jasmine essential oil",
            "yarrow": "chamomile or dried dandelion",
            "vervain": "rosemary or lemon balm",
            "valerian": "chamomile or lavender for calming",
            "elderflower": "chamomile or dried apple blossom",
            "nettle": "dried basil or rosemary",
            "wormwood": "mugwort or sage (much safer alternative)",
            "calendula": "dried chamomile or marigold petals from the garden",
            "dandelion": "dried dandelion from the yard (pesticide-free)",
            "lemon balm": "lemon zest or lemongrass tea",
            "clove": "whole cloves from the spice rack",
            "star anise": "a whole star anise from the spice rack",
            "nutmeg": "a pinch of ground nutmeg from the kitchen",
            "comfrey": "dried chamomile or plantain leaf",
            "echinacea": "dried chamomile or calendula",
            "pine": "a few fresh pine needles from the nearest pine tree",
        }
        return substitutes.get(herb_key, "dried rosemary (universal herbal substitute)")

    @staticmethod
    def _involves_fire(plan: RitualPlan) -> bool:
        """Check if the plan involves fire-related elements."""
        fire_keywords = {"candle", "burn", "flame", "fire", "incense", "smoke", "charcoal"}
        text_pool = " ".join(plan.materials + plan.steps).lower()
        intention_text = plan.intention.lower()
        # If no materials or steps yet, assume fire is likely (candle magick is common)
        if not plan.materials and not plan.steps:
            return True
        return any(kw in text_pool or kw in intention_text for kw in fire_keywords)

    @staticmethod
    def _current_season(month: int) -> str:
        """Return the current season name based on month (Northern Hemisphere)."""
        if month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        elif month in (9, 10, 11):
            return "autumn"
        else:
            return "winter"

    @staticmethod
    def _seasonal_tip(intention: str, month: int) -> str:
        """Return a tip linking the intention to the current season."""
        season_map = {
            (1, 2): "Winter's stillness supports introspective and foundational work. "
                    "This is an excellent time for intention-setting and inner transformation.",
            (3, 4, 5): "Spring's expansive energy amplifies growth, new beginnings, and attraction. "
                       "Plant your intentions now and tend them as the world blooms.",
            (6, 7): "Midsummer's peak energy is ideal for manifestation, celebration, and powerful workings. "
                    "Your magick is at maximum strength.",
            (8, 9): "Harvest energy supports gratitude, reaping what you have sown, and abundance work. "
                    "Assess your progress and celebrate your gains.",
            (10, 11): "The thinning veil supports divination, ancestor work, and deep transformation. "
                      "Turn inward and honor what has passed.",
            (12,): "The return of the light supports hope, renewal, and hearth-centered magick. "
                   "Rest, reflect, and kindle your inner flame.",
        }
        for months, tip in season_map.items():
            if month in months:
                return tip
        return "The Wheel turns, and every season supports the practitioner who aligns with its energy."

    @staticmethod
    def _seasonal_adjustments(intention: str, month: int) -> list[str]:
        """Return season-specific adjustments for the practice."""
        season_adjustments = {
            "winter": [
                "Use warming herbs like cinnamon and ginger to counter cold energy.",
                "Practice indoors by candlelight for a cozy, focused atmosphere.",
                "Incorporate Yule or Imbolc themes if they align with your intention.",
                "Hot tea or cider as a ritual beverage to ground after the working.",
                "Focus on inner work, shadow work, and gestation of intentions.",
            ],
            "spring": [
                "Incorporate fresh flowers and sprouted seeds on your altar.",
                "Practice outdoors if possible to connect with the season's expansive energy.",
                "Plant literal seeds imbued with your intention.",
                "Use lighter, floral herbs like lavender, chamomile, and jasmine.",
                "Channel Ostara and Beltane energy for growth and renewal.",
            ],
            "summer": [
                "Practice at dawn or dusk to harness solar transitions.",
                "Use sunflowers, citrine, and gold candles to amplify solar power.",
                "Harvest fresh herbs from your garden at their peak potency.",
                "Incorporate water (swimming, ocean, rivers) for balance and cleansing.",
                "Channel Litha and Lughnasadh energy for manifestation and first harvest.",
            ],
            "autumn": [
                "Use harvest imagery: apples, gourds, fallen leaves, nuts, and seeds.",
                "Practice gratitude rituals alongside your intention.",
                "Incorporate root vegetables, warm spices, and apple cider on your altar.",
                "This season supports release work, banishing, and transformation.",
                "Channel Mabon and Samhain energy for reflection and ancestor connection.",
            ],
        }
        if month in (3, 4, 5):
            return season_adjustments["spring"]
        elif month in (6, 7, 8):
            return season_adjustments["summer"]
        elif month in (9, 10, 11):
            return season_adjustments["autumn"]
        else:
            return season_adjustments["winter"]

    @staticmethod
    def _best_time_of_day(primary_intention: str) -> str:
        """Return the best time of day for a given intention."""
        time_map = {
            "protection": "Dusk or midnight, when protective energies are strongest at thresholds.",
            "love": "Dawn, as the world awakens and new energy emerges, or Friday evening.",
            "prosperity": "Noon, when the sun (and solar energy of abundance) is at its peak.",
            "healing": "Dawn or the hour just before sleep, when the body is most receptive.",
            "divination": "Midnight, the witching hour, or any liminal time (dawn, dusk).",
            "banishing": "Midnight or the last hour before dawn, as darkness begins to recede.",
            "cleansing": "Dawn, to begin fresh with the new day's clean energy.",
            "creativity": "Mid-morning, when mental energy is high and the sun is rising.",
            "wisdom": "Dusk, the contemplative hour between day and night.",
            "confidence": "Noon, when solar energy and personal power are at their zenith.",
            "communication": "Mid-morning, when the mind is sharp and Mercury's influence is strong.",
            "grounding": "Midnight, when earth energy is deepest, or barefoot at dawn.",
            "transformation": "Midnight, the turning point when one day becomes the next.",
            "peace": "Dusk, as the world settles and calming energy prevails.",
            "courage": "Dawn, when light conquers darkness and the day begins with determination.",
        }
        return time_map.get(primary_intention,
                           "Dawn or dusk, the liminal hours that amplify all workings.")

    @staticmethod
    def _signs_of_success(intention: str) -> list[str]:
        """Return signs that the magick is manifesting for a given intention."""
        intention_lower = intention.lower()
        signs_map = {
            "protection": [
                "A feeling of calm security and reduced anxiety.",
                "Negative people or situations naturally distance themselves.",
                "You notice protective symbols (shields, guardians) in dreams or daily life.",
                "Conflicts resolve more easily than expected.",
                "A sense that you have more energetic space and breathing room.",
            ],
            "love": [
                "Increased warmth and openness in your existing relationships.",
                "New people entering your life who feel aligned and genuine.",
                "Heightened self-love, improved self-image, and confidence.",
                "Dreaming of love, roses, or partnership imagery.",
                "Feeling magnetically attractive or noticing others drawn to you.",
            ],
            "prosperity": [
                "Unexpected money arriving: gifts, refunds, found cash, opportunities.",
                "New job leads, client inquiries, or business ideas surfacing.",
                "A shift in your money mindset from scarcity to abundance.",
                "Feeling generous and abundant even before the money arrives.",
                "Dreaming of gold, green, or overflowing containers.",
            ],
            "healing": [
                "Gradual improvement in physical symptoms or energy levels.",
                "Emotional breakthroughs, crying that feels cleansing rather than painful.",
                "Better sleep and more vivid, healing dreams.",
                "A desire to eat nourishing foods and care for your body.",
                "Feeling lighter, more hopeful, and more present.",
            ],
            "divination": [
                "Heightened intuition in daily decisions.",
                "More vivid, symbolic, or prophetic dreams.",
                "Seeing repeating numbers, symbols, or patterns (synchronicities).",
                "Feeling drawn to particular cards, runes, or oracles.",
                "Receiving clear messages through meditation or quiet reflection.",
            ],
            "banishing": [
                "The unwanted influence weakening or disappearing entirely.",
                "A feeling of relief, lightness, or freedom.",
                "Physical sensations of release: deep sighs, yawning, chills.",
                "The problematic person or pattern fading from your life.",
                "Dreaming of sweeping, cleaning, or water washing things away.",
            ],
        }
        for keyword, signs in signs_map.items():
            if keyword in intention_lower:
                return signs
        # Generic signs
        return [
            "Synchronicities related to your intention appearing in daily life.",
            "A subtle but persistent feeling that things are shifting in the right direction.",
            "Dreams related to your intention or its symbols.",
            "Changes in your own behavior, perspective, or confidence.",
            "Opportunities appearing that align with what you asked for.",
        ]

    @staticmethod
    def _calculate_alignment(plan: RitualPlan, primary: str) -> dict:
        """Calculate a 0-100 alignment score with breakdown."""
        score = 0
        breakdown = {}

        # Intention clarity (20 points)
        if plan.intention and len(plan.intention.strip()) > 2:
            intention_score = 15
            if primary in INTENTION_MAP:
                intention_score = 20
            breakdown["intention_clarity"] = intention_score
            score += intention_score
        else:
            breakdown["intention_clarity"] = 0

        # Correspondence match (20 points)
        enrichments = plan.enrichments
        corr_score = 0
        if enrichments.get("herbs"):
            corr_score += 5
        if enrichments.get("crystals"):
            corr_score += 5
        if enrichments.get("colors"):
            corr_score += 5
        if enrichments.get("elements"):
            corr_score += 5
        breakdown["correspondence_match"] = corr_score
        score += corr_score

        # Timing awareness (20 points)
        timing_score = 0
        if plan.moon_phase:
            optimal_phase = _OPTIMAL_MOON_PHASE.get(primary, ("full_moon",))[0]
            if plan.moon_phase.lower().replace(" ", "_") == optimal_phase:
                timing_score += 10
            else:
                timing_score += 5  # at least they considered timing
        else:
            timing_score += 5  # timing guidance is provided even if not pre-set
        if plan.timing:
            timing_score += 10
        else:
            timing_score += 5  # optimization provides timing
        breakdown["timing_awareness"] = timing_score
        score += timing_score

        # Safety and ethics (20 points)
        safety_score = 0
        if plan.fortifications.get("fire_safety"):
            safety_score += 5
        if plan.fortifications.get("herb_safety"):
            safety_score += 5
        if plan.fortifications.get("ethical_notes"):
            safety_score += 5
        if plan.fortifications.get("medical_disclaimer"):
            safety_score += 5
        breakdown["safety_and_ethics"] = safety_score
        score += safety_score

        # Completeness (20 points)
        completeness_score = 0
        if plan.expansions.get("beginner_version"):
            completeness_score += 5
        if plan.expansions.get("intermediate_version"):
            completeness_score += 5
        if plan.anticipations.get("aftercare"):
            completeness_score += 5
        if plan.anticipations.get("preparation_checklist"):
            completeness_score += 5
        breakdown["completeness"] = completeness_score
        score += completeness_score

        return {
            "total_score": min(score, 100),
            "breakdown": breakdown,
            "interpretation": (
                "Excellent alignment"
                if score >= 90
                else "Strong alignment"
                if score >= 75
                else "Good alignment with room to improve"
                if score >= 60
                else "Moderate alignment — review timing and correspondences"
                if score >= 40
                else "Low alignment — consider adjusting your approach"
            ),
        }

    @staticmethod
    def _seasonal_alignment(intention: str, month: int) -> str:
        """Describe how the current season helps or hinders the intention."""
        intention_lower = intention.lower()

        # Spring (growth, attraction)
        if month in (3, 4, 5):
            boosted = ["love", "creativity", "prosperity", "healing", "confidence"]
            if any(k in intention_lower for k in boosted):
                return "Strongly aligned. Spring's expansive, growth-oriented energy naturally amplifies this intention."
            if any(k in intention_lower for k in ("banishing", "transformation")):
                return "Moderately aligned. Banishing and transformation can work in spring by clearing space for new growth."
            return "Aligned. Spring supports most workings through its energy of renewal and emergence."

        # Summer (peak power, manifestation)
        if month in (6, 7, 8):
            boosted = ["confidence", "creativity", "prosperity", "courage", "communication"]
            if any(k in intention_lower for k in boosted):
                return "Powerfully aligned. Summer's peak solar energy supercharges action-oriented intentions."
            if any(k in intention_lower for k in ("peace", "grounding", "healing")):
                return "Moderately aligned. Add grounding elements to balance summer's intense outward energy."
            return "Aligned. Summer's abundant energy supports all workings performed with focus."

        # Autumn (release, reflection, harvest)
        if month in (9, 10, 11):
            boosted = ["banishing", "transformation", "divination", "grounding", "wisdom"]
            if any(k in intention_lower for k in boosted):
                return "Deeply aligned. Autumn's reflective, releasing energy powerfully supports this intention."
            if any(k in intention_lower for k in ("love", "creativity", "prosperity")):
                return "Moderately aligned. Frame these intentions as harvesting what you have already planted."
            return "Aligned. The thinning veil and harvest energy support inward-focused workings."

        # Winter (rest, gestation, inner work)
        if month in (12, 1, 2):
            boosted = ["protection", "healing", "transformation", "divination", "peace", "grounding"]
            if any(k in intention_lower for k in boosted):
                return "Strongly aligned. Winter's introspective stillness amplifies this inner-focused intention."
            if any(k in intention_lower for k in ("confidence", "creativity", "prosperity")):
                return "Moderately aligned. Winter is a time of gestation; plant these seeds now for spring emergence."
            return "Aligned. Winter's quiet power supports the practitioner who works with patience and depth."

        return "Aligned. The Wheel of the Year supports all sincere practice."
