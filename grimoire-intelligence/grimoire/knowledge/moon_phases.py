"""
Lunar calculations and magical correspondences.

Provides moon phase data, zodiac sign correspondences, and simplified
astronomical calculations for use in grimoire intelligence systems.
All correspondences follow Wiccan and traditional witchcraft traditions.
"""

import math
import datetime
import calendar


# ---------------------------------------------------------------------------
# Moon Phase Correspondences
# ---------------------------------------------------------------------------

MOON_PHASES: dict[str, dict] = {
    "new_moon": {
        "name": "New Moon",
        "emoji": "\U0001f311",
        "illumination_range": [0.0, 0.03],
        "magical_energy": "receptive, introspective, deeply still",
        "best_for": [
            "new beginnings",
            "setting intentions",
            "divination",
            "shadow work",
            "rest and reflection",
            "banishing and binding",
            "breaking bad habits",
        ],
        "avoid": [
            "completion spells",
            "harvest rituals",
            "charging crystals in moonlight",
            "spells requiring outward energy",
        ],
        "element": "earth",
        "keywords": ["seed", "potential", "darkness", "stillness", "void", "rebirth"],
        "herbs": ["comfrey", "mugwort", "patchouli", "black cohosh", "poppy"],
        "crystals": ["obsidian", "black tourmaline", "labradorite", "smoky quartz"],
        "colors": ["black", "dark purple", "silver"],
        "meditation_focus": "Turn inward and sit with the fertile darkness. "
        "What seeds do you wish to plant in this cycle?",
        "journal_prompt": "What intention am I ready to set for this lunar cycle? "
        "What must I release to make room for it?",
        "daily_guidance": "Honor the darkness today. Rest, reflect, and write down "
        "one clear intention for the coming cycle.",
    },
    "waxing_crescent": {
        "name": "Waxing Crescent",
        "emoji": "\U0001f312",
        "illumination_range": [0.03, 0.25],
        "magical_energy": "emerging, hopeful, gently building",
        "best_for": [
            "attraction spells",
            "courage and confidence magick",
            "new projects",
            "making plans",
            "animal magick",
            "setting foundations",
        ],
        "avoid": [
            "banishing work",
            "aggressive protection spells",
            "endings",
        ],
        "element": "air",
        "keywords": ["sprout", "hope", "emergence", "curiosity", "wish", "faith"],
        "herbs": ["lavender", "jasmine", "lemon balm", "chamomile", "angelica"],
        "crystals": ["citrine", "aventurine", "rose quartz", "amazonite"],
        "colors": ["white", "light green", "pale yellow"],
        "meditation_focus": "Visualize your intention as a tiny green shoot breaking "
        "through the soil. Nurture it with breath and belief.",
        "journal_prompt": "What first step can I take today toward my new moon intention? "
        "What fears arise, and how can I soothe them?",
        "daily_guidance": "Take one small action toward your goal. The crescent "
        "rewards initiative, however modest.",
    },
    "first_quarter": {
        "name": "First Quarter",
        "emoji": "\U0001f313",
        "illumination_range": [0.25, 0.50],
        "magical_energy": "active, decisive, challenged",
        "best_for": [
            "strength and courage spells",
            "overcoming obstacles",
            "decision-making rituals",
            "motivation magick",
            "protection magick",
            "creative projects",
        ],
        "avoid": [
            "passive workings",
            "meditation-only rituals",
            "surrender spells",
        ],
        "element": "fire",
        "keywords": ["action", "challenge", "commitment", "momentum", "struggle", "will"],
        "herbs": ["basil", "cinnamon", "ginger", "thistle", "nettle", "bay laurel"],
        "crystals": ["carnelian", "red jasper", "tiger's eye", "pyrite"],
        "colors": ["red", "orange", "gold"],
        "meditation_focus": "Feel the fire of determination in your solar plexus. "
        "Challenges are not walls but whetstones for your will.",
        "journal_prompt": "What obstacle stands between me and my intention? "
        "What strength do I already possess to overcome it?",
        "daily_guidance": "Face a challenge head-on today. The half-lit moon "
        "asks you to choose action over hesitation.",
    },
    "waxing_gibbous": {
        "name": "Waxing Gibbous",
        "emoji": "\U0001f314",
        "illumination_range": [0.50, 0.97],
        "magical_energy": "refining, adjusting, intensifying",
        "best_for": [
            "refining spells already cast",
            "patience magick",
            "glamour and beauty spells",
            "health and healing rituals",
            "editing and perfecting creative works",
            "building energy for full moon",
        ],
        "avoid": [
            "starting brand-new workings",
            "drastic changes",
            "impulsive magick",
        ],
        "element": "water",
        "keywords": ["refinement", "patience", "trust", "adjustment", "anticipation", "polish"],
        "herbs": ["rose", "vervain", "yarrow", "elderflower", "meadowsweet"],
        "crystals": ["moonstone", "clear quartz", "selenite", "amethyst"],
        "colors": ["light blue", "silver", "lavender"],
        "meditation_focus": "Like a sculptor removing excess stone, refine your "
        "vision. Trust that the fullness is approaching.",
        "journal_prompt": "What adjustments does my intention need? "
        "Where can I practice patience and trust the process?",
        "daily_guidance": "Fine-tune rather than start fresh. Review your progress "
        "and make small, intentional corrections.",
    },
    "full_moon": {
        "name": "Full Moon",
        "emoji": "\U0001f315",
        "illumination_range": [0.97, 1.0],
        "magical_energy": "peak, illuminated, powerful, abundant",
        "best_for": [
            "charging tools and crystals",
            "divination and scrying",
            "love spells",
            "abundance and prosperity magick",
            "psychic development",
            "healing rituals",
            "completion spells",
            "full moon water",
            "gratitude rituals",
        ],
        "avoid": [
            "banishing (save for waning)",
            "cursework",
            "spells requiring subtlety",
        ],
        "element": "spirit",
        "keywords": ["fullness", "illumination", "power", "completion", "celebration", "clarity"],
        "herbs": ["jasmine", "sandalwood", "lotus", "white rose", "frankincense", "mugwort"],
        "crystals": ["moonstone", "selenite", "clear quartz", "opal", "pearl"],
        "colors": ["white", "silver", "pale gold", "iridescent"],
        "meditation_focus": "Bathe in the full radiance of the moon. Everything "
        "is illuminated; see clearly and celebrate how far you have come.",
        "journal_prompt": "What has come to fruition this cycle? "
        "What do I see clearly now that was hidden before?",
        "daily_guidance": "Celebrate and give thanks. Charge your crystals and "
        "tools. Your magick is at its peak tonight.",
    },
    "waning_gibbous": {
        "name": "Waning Gibbous",
        "emoji": "\U0001f316",
        "illumination_range": [0.50, 0.97],
        "magical_energy": "grateful, sharing, introspective",
        "best_for": [
            "gratitude magick",
            "sharing knowledge",
            "teaching rituals",
            "breaking bad habits",
            "cord-cutting",
            "cleansing and purification",
            "returning favors to spirits",
        ],
        "avoid": [
            "new attraction spells",
            "beginning new projects",
            "growth magick",
        ],
        "element": "water",
        "keywords": ["gratitude", "wisdom", "sharing", "dissemination", "teaching", "harvest"],
        "herbs": ["sage", "thyme", "hyssop", "eucalyptus", "rosemary"],
        "crystals": ["lapis lazuli", "sodalite", "blue lace agate", "fluorite"],
        "colors": ["blue", "grey", "muted purple"],
        "meditation_focus": "Reflect on the gifts this cycle has brought. "
        "What wisdom can you share with others?",
        "journal_prompt": "What lessons has this lunar cycle taught me? "
        "How can I share what I have learned?",
        "daily_guidance": "Give back today. Share a skill, a kind word, "
        "or an offering to the spirits who have aided you.",
    },
    "last_quarter": {
        "name": "Last Quarter",
        "emoji": "\U0001f317",
        "illumination_range": [0.25, 0.50],
        "magical_energy": "releasing, forgiving, clearing",
        "best_for": [
            "banishing spells",
            "breaking hexes",
            "forgiveness rituals",
            "releasing attachments",
            "clearing clutter (physical and energetic)",
            "justice spells",
            "addiction-breaking magick",
        ],
        "avoid": [
            "new beginnings",
            "attraction magick",
            "building energy",
        ],
        "element": "earth",
        "keywords": ["release", "forgiveness", "clearing", "letting go", "closure", "justice"],
        "herbs": ["rue", "agrimony", "black salt", "garlic", "wormwood", "lemon"],
        "crystals": ["apache tear", "jet", "black obsidian", "hematite"],
        "colors": ["dark grey", "black", "deep indigo"],
        "meditation_focus": "Hold what you wish to release in your open palms. "
        "Exhale and let it dissolve into the fading moonlight.",
        "journal_prompt": "What am I ready to forgive or release? "
        "What no longer serves my highest good?",
        "daily_guidance": "Let something go today, whether a grudge, a habit, "
        "or clutter in your space. Lighten your load.",
    },
    "waning_crescent": {
        "name": "Waning Crescent",
        "emoji": "\U0001f318",
        "illumination_range": [0.0, 0.25],
        "magical_energy": "surrendering, resting, composting",
        "best_for": [
            "rest and recuperation",
            "dreamwork and lucid dreaming",
            "ancestor communication",
            "past-life work",
            "meditation",
            "preparing for the new cycle",
            "prophecy and vision quests",
        ],
        "avoid": [
            "any high-energy spellwork",
            "manifesting or attraction",
            "charging objects",
        ],
        "element": "air",
        "keywords": ["surrender", "rest", "dreams", "ancestors", "composting", "liminal"],
        "herbs": ["valerian", "passionflower", "skullcap", "myrrh", "cypress"],
        "crystals": ["lepidolite", "howlite", "celestite", "ametrine"],
        "colors": ["dark blue", "charcoal", "soft grey"],
        "meditation_focus": "Sink into the liminal space between endings and beginnings. "
        "Listen for the whisper of what comes next.",
        "journal_prompt": "What dreams or visions have visited me? "
        "What is composting in the dark, readying itself to sprout?",
        "daily_guidance": "Rest deeply. This is the balsamic moon, the dark before "
        "the dawn. Honour your need for stillness.",
    },
}


# ---------------------------------------------------------------------------
# Moon-in-Zodiac-Sign Correspondences
# ---------------------------------------------------------------------------

MOON_IN_SIGNS: dict[str, dict] = {
    "aries": {
        "sign": "Aries",
        "element": "fire",
        "quality": "cardinal",
        "energy": "bold, impulsive, pioneering, courageous",
        "best_for": [
            "spells for courage and confidence",
            "initiating new ventures",
            "competitive success",
            "physical vitality",
            "breaking through stagnation",
        ],
        "avoid": ["patience-requiring rituals", "diplomacy magick", "passive workings"],
        "herbs": ["cayenne", "dragon's blood", "ginger", "nettle", "allspice"],
        "crystals": ["carnelian", "red jasper", "garnet", "bloodstone"],
        "body_area": "head",
    },
    "taurus": {
        "sign": "Taurus",
        "element": "earth",
        "quality": "fixed",
        "energy": "stable, sensual, grounded, abundant",
        "best_for": [
            "money and prosperity spells",
            "garden magick",
            "cooking and kitchen witchery",
            "self-worth rituals",
            "grounding and stability spells",
        ],
        "avoid": ["change and transformation spells", "speed magick", "travel spells"],
        "herbs": ["mint", "thyme", "apple blossom", "patchouli", "rose"],
        "crystals": ["emerald", "rose quartz", "jade", "malachite"],
        "body_area": "throat and neck",
    },
    "gemini": {
        "sign": "Gemini",
        "element": "air",
        "quality": "mutable",
        "energy": "communicative, curious, versatile, witty",
        "best_for": [
            "communication spells",
            "studying and learning magick",
            "writing rituals",
            "travel magick",
            "networking and social connection",
        ],
        "avoid": ["long-term commitment spells", "deep emotional work", "grounding rituals"],
        "herbs": ["lavender", "lemongrass", "parsley", "dill", "caraway"],
        "crystals": ["agate", "citrine", "blue lace agate", "aquamarine"],
        "body_area": "arms, hands, and lungs",
    },
    "cancer": {
        "sign": "Cancer",
        "element": "water",
        "quality": "cardinal",
        "energy": "nurturing, emotional, protective, intuitive",
        "best_for": [
            "home and hearth magick",
            "family protection spells",
            "emotional healing",
            "kitchen witchery",
            "ancestor work",
            "fertility magick",
        ],
        "avoid": ["emotional detachment spells", "aggressive magick", "public-facing rituals"],
        "herbs": ["chamomile", "lemon balm", "jasmine", "water lily", "moonwort"],
        "crystals": ["moonstone", "pearl", "selenite", "chalcedony"],
        "body_area": "chest and stomach",
    },
    "leo": {
        "sign": "Leo",
        "element": "fire",
        "quality": "fixed",
        "energy": "dramatic, creative, confident, radiant",
        "best_for": [
            "glamour and beauty spells",
            "creative inspiration",
            "leadership magick",
            "performance success",
            "courage and self-expression",
            "love and romance",
        ],
        "avoid": ["humility rituals", "shadow work", "invisibility or hiding spells"],
        "herbs": ["sunflower", "frankincense", "cinnamon", "saffron", "bay laurel"],
        "crystals": ["sunstone", "tiger's eye", "amber", "golden topaz"],
        "body_area": "heart and upper back",
    },
    "virgo": {
        "sign": "Virgo",
        "element": "earth",
        "quality": "mutable",
        "energy": "analytical, healing, practical, purifying",
        "best_for": [
            "health and healing spells",
            "purification and cleansing",
            "organization magick",
            "herbal preparations",
            "detail-oriented workings",
            "service and devotion rituals",
        ],
        "avoid": ["big-picture visioning", "chaos magick", "spontaneous ritual"],
        "herbs": ["lavender", "fennel", "valerian", "marjoram", "witch hazel"],
        "crystals": ["peridot", "amazonite", "moss agate", "sapphire"],
        "body_area": "digestive system",
    },
    "libra": {
        "sign": "Libra",
        "element": "air",
        "quality": "cardinal",
        "energy": "harmonizing, diplomatic, aesthetic, balanced",
        "best_for": [
            "love and partnership spells",
            "justice and legal magick",
            "beauty rituals",
            "harmony and peace spells",
            "artistic inspiration",
            "balance and equilibrium workings",
        ],
        "avoid": ["solitary or selfish workings", "aggressive magick", "decision-forcing spells"],
        "herbs": ["rose", "violet", "catnip", "pennyroyal", "bergamot"],
        "crystals": ["rose quartz", "lapis lazuli", "opal", "kunzite"],
        "body_area": "kidneys and lower back",
    },
    "scorpio": {
        "sign": "Scorpio",
        "element": "water",
        "quality": "fixed",
        "energy": "intense, transformative, secretive, powerful",
        "best_for": [
            "transformation and rebirth spells",
            "sex magick",
            "death and ancestor work",
            "deep divination",
            "psychic development",
            "shadow work",
            "uncovering hidden truths",
        ],
        "avoid": ["lighthearted or surface-level workings", "trust spells", "transparency rituals"],
        "herbs": ["wormwood", "basil", "blackthorn", "dragon's blood", "damiana"],
        "crystals": ["obsidian", "garnet", "malachite", "black tourmaline"],
        "body_area": "reproductive organs",
    },
    "sagittarius": {
        "sign": "Sagittarius",
        "element": "fire",
        "quality": "mutable",
        "energy": "expansive, adventurous, philosophical, optimistic",
        "best_for": [
            "travel and adventure spells",
            "luck and fortune magick",
            "higher learning rituals",
            "spiritual expansion",
            "freedom and liberation spells",
            "philosophical inquiry",
        ],
        "avoid": ["restriction spells", "detail work", "binding magick"],
        "herbs": ["sage", "star anise", "clove", "dandelion", "cedar"],
        "crystals": ["turquoise", "lapis lazuli", "sodalite", "amethyst"],
        "body_area": "hips and thighs",
    },
    "capricorn": {
        "sign": "Capricorn",
        "element": "earth",
        "quality": "cardinal",
        "energy": "disciplined, ambitious, structured, enduring",
        "best_for": [
            "career and ambition spells",
            "long-term goal magick",
            "discipline and structure rituals",
            "ancestral authority work",
            "building foundations",
            "business success",
        ],
        "avoid": ["spontaneity spells", "emotional vulnerability work", "playfulness rituals"],
        "herbs": ["comfrey", "horsetail", "thyme", "patchouli", "slippery elm"],
        "crystals": ["garnet", "onyx", "jet", "smoky quartz"],
        "body_area": "knees, bones, and joints",
    },
    "aquarius": {
        "sign": "Aquarius",
        "element": "air",
        "quality": "fixed",
        "energy": "innovative, humanitarian, eccentric, visionary",
        "best_for": [
            "community and group magick",
            "innovation and invention spells",
            "breaking free from limitation",
            "social justice rituals",
            "friendship magick",
            "technology blessings",
        ],
        "avoid": ["tradition-bound rituals", "emotional intimacy spells", "conformity workings"],
        "herbs": ["star anise", "frankincense", "lavender", "orchid", "valerian"],
        "crystals": ["amethyst", "aquamarine", "fluorite", "labradorite"],
        "body_area": "ankles and circulatory system",
    },
    "pisces": {
        "sign": "Pisces",
        "element": "water",
        "quality": "mutable",
        "energy": "dreamy, psychic, compassionate, transcendent",
        "best_for": [
            "psychic development",
            "dreamwork and lucid dreaming",
            "compassion and empathy rituals",
            "artistic and musical magick",
            "spiritual healing",
            "past-life work",
            "water magick",
        ],
        "avoid": ["logical or analytical workings", "boundary-setting spells", "grounding rituals"],
        "herbs": ["mugwort", "seaweed", "lotus", "willow", "water lily"],
        "crystals": ["amethyst", "aquamarine", "fluorite", "celestite", "moonstone"],
        "body_area": "feet and lymphatic system",
    },
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def get_moon_phase_from_illumination(illumination: float, is_waxing: bool) -> str:
    """Return the moon phase key based on illumination fraction and waxing/waning.

    Args:
        illumination: Float between 0.0 and 1.0 representing the fraction
            of the moon's disk that is illuminated.
        is_waxing: True if the moon is gaining light, False if losing.

    Returns:
        A key from MOON_PHASES (e.g. ``"waxing_crescent"``).
    """
    illumination = max(0.0, min(1.0, illumination))

    if illumination >= 0.97:
        return "full_moon"
    if illumination <= 0.03:
        return "new_moon"

    if is_waxing:
        if illumination < 0.25:
            return "waxing_crescent"
        if illumination < 0.50:
            return "first_quarter"
        return "waxing_gibbous"
    else:
        if illumination < 0.25:
            return "waning_crescent"
        if illumination < 0.50:
            return "last_quarter"
        return "waning_gibbous"


def get_phase_data(phase_name: str) -> dict | None:
    """Return the full correspondence dict for a moon phase.

    Args:
        phase_name: A key such as ``"full_moon"`` or ``"waxing_crescent"``.

    Returns:
        The correspondence dict, or ``None`` if the phase name is unknown.
    """
    return MOON_PHASES.get(phase_name.lower().replace(" ", "_"))


def get_sign_data(sign: str) -> dict | None:
    """Return the full correspondence dict for a zodiac sign.

    Args:
        sign: A sign name such as ``"Aries"`` or ``"aries"``.

    Returns:
        The correspondence dict, or ``None`` if the sign name is unknown.
    """
    return MOON_IN_SIGNS.get(sign.lower().strip())


def get_combined_energy(phase: str, sign: str) -> dict:
    """Combine moon phase and zodiac sign energies into a unified reading.

    Args:
        phase: A moon phase key (e.g. ``"full_moon"``).
        sign: A zodiac sign name (e.g. ``"scorpio"``).

    Returns:
        A dict with keys ``phase``, ``sign``, ``combined_best_for``,
        ``combined_avoid``, ``combined_herbs``, ``combined_crystals``,
        ``combined_elements``, and ``summary``.

    Raises:
        ValueError: If either phase or sign is unknown.
    """
    phase_data = get_phase_data(phase)
    sign_data = get_sign_data(sign)

    if phase_data is None:
        raise ValueError(f"Unknown moon phase: {phase!r}")
    if sign_data is None:
        raise ValueError(f"Unknown zodiac sign: {sign!r}")

    # Merge lists, preserving order and removing duplicates
    def _merge_lists(a: list, b: list) -> list:
        seen: set[str] = set()
        merged: list[str] = []
        for item in a + b:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    combined_best = _merge_lists(phase_data["best_for"], sign_data["best_for"])
    combined_avoid = _merge_lists(phase_data["avoid"], sign_data["avoid"])
    combined_herbs = _merge_lists(phase_data["herbs"], sign_data["herbs"])
    combined_crystals = _merge_lists(phase_data["crystals"], sign_data["crystals"])

    elements = []
    if phase_data["element"] not in elements:
        elements.append(phase_data["element"])
    if sign_data["element"] not in elements:
        elements.append(sign_data["element"])

    summary = (
        f"The {phase_data['name']} in {sign_data['sign']} blends "
        f"{phase_data['magical_energy']} energy with the "
        f"{sign_data['energy']} nature of {sign_data['sign']}. "
        f"Elements at play: {' and '.join(elements)}."
    )

    return {
        "phase": phase_data,
        "sign": sign_data,
        "combined_best_for": combined_best,
        "combined_avoid": combined_avoid,
        "combined_herbs": combined_herbs,
        "combined_crystals": combined_crystals,
        "combined_elements": elements,
        "summary": summary,
    }


def calculate_moon_phase(year: int, month: int, day: int) -> tuple[str, float]:
    """Calculate an approximate moon phase for a given date.

    Uses a modified Conway / Trig algorithm to estimate the age of the moon
    (days since last new moon) and derive illumination and phase.  This is a
    reasonable approximation good to roughly +/- 1 day; it is NOT intended for
    astronomical precision.

    Args:
        year: Calendar year (e.g. 2026).
        month: Month number (1-12).
        day: Day of the month (1-31).

    Returns:
        A tuple of ``(phase_key, approximate_illumination)`` where
        ``phase_key`` is a key from :data:`MOON_PHASES` and illumination
        is a float between 0.0 and 1.0.
    """
    # --- Julian Day Number (simplified, valid for Gregorian dates) ---
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (
        day
        + (153 * m + 2) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )

    # Known new moon epoch: 2000-01-06 18:14 UTC  (JDN 2451551.26)
    known_new_moon_jdn = 2451550.1 + 0.76  # 2451550.86 ~ Jan 6 2000 ~18h UT

    # Synodic month (mean lunation period)
    synodic_month = 29.53058867

    # Days since the known new moon
    days_since = jdn - known_new_moon_jdn

    # Moon age in current cycle (0 = new moon, ~14.76 = full moon)
    moon_age = days_since % synodic_month

    # Illumination approximation using cosine
    # At age 0 -> illumination 0, age ~14.76 -> illumination 1.0
    phase_angle = (moon_age / synodic_month) * 2.0 * math.pi
    illumination = (1.0 - math.cos(phase_angle)) / 2.0
    illumination = max(0.0, min(1.0, illumination))

    # Determine waxing / waning
    is_waxing = moon_age < (synodic_month / 2.0)

    phase_key = get_moon_phase_from_illumination(illumination, is_waxing)

    return phase_key, round(illumination, 4)


def get_lunar_month_forecast(year: int, month: int) -> list[dict]:
    """Return approximate moon phase info for every day of a calendar month.

    Args:
        year: Calendar year.
        month: Month number (1-12).

    Returns:
        A list of dicts, one per day, each containing:

        - ``date`` (``datetime.date``)
        - ``day`` (int)
        - ``phase_key`` (str)
        - ``phase_name`` (str)
        - ``illumination`` (float, 0.0-1.0)
        - ``emoji`` (str)
        - ``magical_energy`` (str)
        - ``best_for`` (list[str])
        - ``daily_guidance`` (str)
    """
    num_days = calendar.monthrange(year, month)[1]
    forecast: list[dict] = []

    for day in range(1, num_days + 1):
        phase_key, illumination = calculate_moon_phase(year, month, day)
        phase_data = MOON_PHASES[phase_key]

        forecast.append(
            {
                "date": datetime.date(year, month, day),
                "day": day,
                "phase_key": phase_key,
                "phase_name": phase_data["name"],
                "illumination": illumination,
                "emoji": phase_data["emoji"],
                "magical_energy": phase_data["magical_energy"],
                "best_for": phase_data["best_for"],
                "daily_guidance": phase_data["daily_guidance"],
            }
        )

    return forecast
