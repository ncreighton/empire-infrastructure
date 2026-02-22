"""Planetary hour calculations and magical correspondences.

Implements the traditional Chaldean system of planetary hours used in
Western ceremonial magic, horary astrology, and folk magical traditions.
Each day is divided into 12 day hours (sunrise to sunset) and 12 night
hours (sunset to next sunrise), each ruled by a planet in the Chaldean
sequence beginning with the day's ruler.
"""

from datetime import datetime
import math

# --------------------------------------------------------------------------- #
#  Core sequences                                                              #
# --------------------------------------------------------------------------- #

PLANETARY_ORDER: list[str] = [
    "Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon",
]

# Weekday 0 = Monday (Python isoweekday()-1 convention)
DAY_RULERS: dict[int, str] = {
    0: "Moon",       # Monday
    1: "Mars",       # Tuesday
    2: "Mercury",    # Wednesday
    3: "Jupiter",    # Thursday
    4: "Venus",      # Friday
    5: "Saturn",     # Saturday
    6: "Sun",        # Sunday
}

# --------------------------------------------------------------------------- #
#  Correspondences                                                             #
# --------------------------------------------------------------------------- #

PLANET_CORRESPONDENCES: dict[str, dict] = {
    "Sun": {
        "day": "Sunday",
        "metal": "gold",
        "colors": ["gold", "yellow", "orange"],
        "herbs": ["st_johns_wort", "chamomile", "cinnamon", "frankincense"],
        "crystals": ["citrine", "sunstone", "amber", "tiger_eye"],
        "magical_domains": [
            "success", "vitality", "leadership", "fame", "confidence", "health",
        ],
        "zodiac": ["Leo"],
        "number": 6,
        "archangel": "Michael",
        "incense": "frankincense",
    },
    "Moon": {
        "day": "Monday",
        "metal": "silver",
        "colors": ["white", "silver", "pale_blue"],
        "herbs": ["jasmine", "mugwort", "willow", "camphor"],
        "crystals": ["moonstone", "selenite", "pearl", "clear_quartz"],
        "magical_domains": [
            "intuition", "dreams", "fertility", "emotions", "psychic_ability",
            "cleansing",
        ],
        "zodiac": ["Cancer"],
        "number": 9,
        "archangel": "Gabriel",
        "incense": "jasmine",
    },
    "Mars": {
        "day": "Tuesday",
        "metal": "iron",
        "colors": ["red", "scarlet", "crimson"],
        "herbs": ["dragon_blood", "nettle", "garlic", "black_pepper"],
        "crystals": ["bloodstone", "red_jasper", "garnet", "ruby"],
        "magical_domains": [
            "protection", "courage", "strength", "victory", "lust", "banishing",
        ],
        "zodiac": ["Aries", "Scorpio"],
        "number": 5,
        "archangel": "Samael",
        "incense": "dragon_blood",
    },
    "Mercury": {
        "day": "Wednesday",
        "metal": "mercury",
        "colors": ["orange", "violet", "multicolor"],
        "herbs": ["lavender", "fennel", "dill", "mastic"],
        "crystals": ["agate", "fluorite", "opal", "aventurine"],
        "magical_domains": [
            "communication", "intellect", "travel", "divination", "commerce",
            "wisdom",
        ],
        "zodiac": ["Gemini", "Virgo"],
        "number": 8,
        "archangel": "Raphael",
        "incense": "lavender",
    },
    "Jupiter": {
        "day": "Thursday",
        "metal": "tin",
        "colors": ["blue", "purple", "royal_blue"],
        "herbs": ["sage", "hyssop", "nutmeg", "cedar"],
        "crystals": ["amethyst", "lapis_lazuli", "sapphire", "turquoise"],
        "magical_domains": [
            "prosperity", "abundance", "luck", "justice", "wisdom", "expansion",
        ],
        "zodiac": ["Sagittarius", "Pisces"],
        "number": 4,
        "archangel": "Sachiel",
        "incense": "cedar",
    },
    "Venus": {
        "day": "Friday",
        "metal": "copper",
        "colors": ["green", "pink", "rose"],
        "herbs": ["rose", "yarrow", "vervain", "thyme"],
        "crystals": ["rose_quartz", "emerald", "jade", "malachite"],
        "magical_domains": [
            "love", "beauty", "harmony", "art", "pleasure", "friendship",
            "creativity",
        ],
        "zodiac": ["Taurus", "Libra"],
        "number": 7,
        "archangel": "Anael",
        "incense": "rose",
    },
    "Saturn": {
        "day": "Saturday",
        "metal": "lead",
        "colors": ["black", "dark_brown", "indigo"],
        "herbs": ["comfrey", "mullein", "myrrh", "patchouli"],
        "crystals": ["obsidian", "onyx", "jet", "hematite"],
        "magical_domains": [
            "protection", "banishing", "binding", "discipline", "boundaries",
            "transformation", "grounding",
        ],
        "zodiac": ["Capricorn", "Aquarius"],
        "number": 3,
        "archangel": "Cassiel",
        "incense": "myrrh",
    },
}

# --------------------------------------------------------------------------- #
#  Intention mapping                                                           #
# --------------------------------------------------------------------------- #

INTENTION_TO_PLANET: dict[str, list[str]] = {
    "protection":       ["Mars", "Saturn"],
    "love":             ["Venus"],
    "prosperity":       ["Jupiter", "Sun"],
    "healing":          ["Sun", "Moon"],
    "divination":       ["Moon", "Mercury"],
    "banishing":        ["Saturn", "Mars"],
    "cleansing":        ["Moon"],
    "creativity":       ["Venus", "Mercury"],
    "wisdom":           ["Mercury", "Jupiter"],
    "confidence":       ["Sun", "Mars"],
    "communication":    ["Mercury"],
    "grounding":        ["Saturn"],
    "transformation":   ["Saturn"],
    "peace":            ["Moon", "Venus"],
    "courage":          ["Mars", "Sun"],
}

# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _chaldean_index(planet: str) -> int:
    """Return the index of *planet* in the Chaldean order (case-insensitive)."""
    for i, p in enumerate(PLANETARY_ORDER):
        if p.lower() == planet.lower():
            return i
    raise ValueError(f"Unknown planet: {planet}")


def _planet_at_offset(start_planet: str, offset: int) -> str:
    """Return the planet that is *offset* steps after *start_planet* in the
    Chaldean sequence (which repeats cyclically)."""
    idx = _chaldean_index(start_planet)
    return PLANETARY_ORDER[(idx + offset) % 7]


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def get_day_ruler(weekday: int) -> str:
    """Return the ruling planet for *weekday* (0=Monday .. 6=Sunday)."""
    if weekday not in DAY_RULERS:
        raise ValueError(f"weekday must be 0-6, got {weekday}")
    return DAY_RULERS[weekday]


def get_planetary_hour(hour: int, weekday: int, is_daytime: bool = True) -> str:
    """Return which planet rules the given *hour* (0-11) of the day or night
    portion for the specified *weekday*.

    The first daytime hour (hour 0, is_daytime=True) is always the day ruler.
    Subsequent hours follow the Chaldean sequence.
    """
    if not 0 <= hour <= 11:
        raise ValueError(f"hour must be 0-11, got {hour}")
    ruler = get_day_ruler(weekday)
    # Day hours start at offset 0; night hours start at offset 12
    offset = hour if is_daytime else hour + 12
    return _planet_at_offset(ruler, offset)


def get_planet_data(planet: str) -> dict | None:
    """Return the correspondences dict for *planet*, or ``None`` if unknown.

    Lookup is case-insensitive; the canonical key uses title-case.
    """
    key = planet.strip().title()
    return PLANET_CORRESPONDENCES.get(key)


def get_current_planetary_hour(
    weekday: int,
    hour: int,
    sunrise_hour: int = 6,
    sunset_hour: int = 18,
) -> dict:
    """Determine the current planetary hour based on a 24-hour clock value.

    Uses simple equal-hour division:
      - Day hours:   sunrise_hour to sunset_hour  -> 12 equal segments
      - Night hours: sunset_hour  to next sunrise -> 12 equal segments

    Returns a dict with keys: ``planet``, ``hour_number`` (0-11),
    ``is_daytime``, ``hour_start``, ``hour_end``, and ``correspondences``.
    """
    if sunrise_hour >= sunset_hour:
        raise ValueError("sunrise_hour must be less than sunset_hour")

    day_length = sunset_hour - sunrise_hour          # hours of daylight
    night_length = 24 - day_length                   # hours of darkness
    day_segment = day_length / 12.0
    night_segment = night_length / 12.0

    if sunrise_hour <= hour < sunset_hour:
        # Daytime
        is_daytime = True
        elapsed = hour - sunrise_hour
        hour_number = min(int(elapsed / day_segment), 11)
        hour_start = sunrise_hour + hour_number * day_segment
        hour_end = hour_start + day_segment
    else:
        # Nighttime
        is_daytime = False
        if hour >= sunset_hour:
            elapsed = hour - sunset_hour
        else:
            # After midnight, before sunrise
            elapsed = (24 - sunset_hour) + hour
        hour_number = min(int(elapsed / night_segment), 11)
        hour_start_raw = sunset_hour + hour_number * night_segment
        hour_end_raw = hour_start_raw + night_segment
        # Normalise past midnight
        hour_start = hour_start_raw % 24
        hour_end = hour_end_raw % 24

    planet = get_planetary_hour(hour_number, weekday, is_daytime)
    correspondences = get_planet_data(planet) or {}

    return {
        "planet": planet,
        "hour_number": hour_number,
        "is_daytime": is_daytime,
        "hour_start": round(hour_start, 4),
        "hour_end": round(hour_end, 4),
        "correspondences": correspondences,
    }


def get_all_hours_for_day(
    weekday: int,
    sunrise_hour: int = 6,
    sunset_hour: int = 18,
) -> list[dict]:
    """Return all 24 planetary hours for *weekday* with times and
    correspondences.

    Each entry is a dict matching the shape returned by
    :func:`get_current_planetary_hour`.
    """
    if sunrise_hour >= sunset_hour:
        raise ValueError("sunrise_hour must be less than sunset_hour")

    day_length = sunset_hour - sunrise_hour
    night_length = 24 - day_length
    day_segment = day_length / 12.0
    night_segment = night_length / 12.0

    hours: list[dict] = []

    # 12 day hours
    for i in range(12):
        planet = get_planetary_hour(i, weekday, is_daytime=True)
        start = sunrise_hour + i * day_segment
        end = start + day_segment
        hours.append({
            "planet": planet,
            "hour_number": i,
            "is_daytime": True,
            "hour_start": round(start, 4),
            "hour_end": round(end, 4),
            "correspondences": get_planet_data(planet) or {},
        })

    # 12 night hours
    for i in range(12):
        planet = get_planetary_hour(i, weekday, is_daytime=False)
        start_raw = sunset_hour + i * night_segment
        end_raw = start_raw + night_segment
        hours.append({
            "planet": planet,
            "hour_number": i,
            "is_daytime": False,
            "hour_start": round(start_raw % 24, 4),
            "hour_end": round(end_raw % 24, 4),
            "correspondences": get_planet_data(planet) or {},
        })

    return hours


def get_best_planetary_hour(
    intention: str,
    weekday: int,
    sunrise_hour: int = 6,
    sunset_hour: int = 18,
) -> list[dict]:
    """Return the best planetary hours today for *intention*, sorted by
    relevance (primary planet first, then secondary).

    Each returned dict extends the standard hour dict with a ``relevance``
    field (1 = primary match, 2 = secondary match).
    """
    key = intention.strip().lower()
    planets = INTENTION_TO_PLANET.get(key)
    if planets is None:
        # Fallback: check if the intention matches a magical_domain directly
        planets = []
        for pname, pdata in PLANET_CORRESPONDENCES.items():
            if key in pdata.get("magical_domains", []):
                planets.append(pname)
        if not planets:
            return []

    all_hours = get_all_hours_for_day(weekday, sunrise_hour, sunset_hour)
    results: list[dict] = []

    for entry in all_hours:
        planet = entry["planet"]
        if planet in planets:
            relevance = planets.index(planet) + 1
            results.append({**entry, "relevance": relevance})

    results.sort(key=lambda d: (d["relevance"], d["hour_start"]))
    return results
