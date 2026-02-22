"""Number correspondences and numerological calculations."""

NUMBERS = {
    1: {
        "meaning": "New beginnings, independence, leadership, individuality",
        "planet": "Sun",
        "element": "fire",
        "keywords": ["initiation", "self", "pioneer", "ambition", "unity"],
        "tarot": "The Magician",
        "magical_use": "Starting new projects, self-empowerment, independence spells",
        "day": "Sunday",
        "crystal": "citrine",
        "herb": "cinnamon",
    },
    2: {
        "meaning": "Partnership, balance, duality, cooperation, intuition",
        "planet": "Moon",
        "element": "water",
        "keywords": ["partnership", "balance", "diplomacy", "receptivity", "duality"],
        "tarot": "The High Priestess",
        "magical_use": "Relationship magick, finding balance, enhancing intuition",
        "day": "Monday",
        "crystal": "moonstone",
        "herb": "jasmine",
    },
    3: {
        "meaning": "Creativity, expression, growth, the Triple Goddess",
        "planet": "Jupiter",
        "element": "fire",
        "keywords": ["creativity", "expression", "joy", "expansion", "trinity"],
        "tarot": "The Empress",
        "magical_use": "Creative endeavors, fertility, self-expression, celebration",
        "day": "Thursday",
        "crystal": "amethyst",
        "herb": "lavender",
    },
    4: {
        "meaning": "Stability, foundation, structure, the four elements/directions",
        "planet": "Saturn",
        "element": "earth",
        "keywords": ["foundation", "stability", "order", "structure", "protection"],
        "tarot": "The Emperor",
        "magical_use": "Protection wards, building foundations, stability spells",
        "day": "Saturday",
        "crystal": "hematite",
        "herb": "sage",
    },
    5: {
        "meaning": "Change, freedom, adventure, the five points of the pentacle",
        "planet": "Mercury",
        "element": "air",
        "keywords": ["change", "freedom", "adventure", "curiosity", "spirit"],
        "tarot": "The Hierophant",
        "magical_use": "Transformation, travel protection, breaking stagnation",
        "day": "Wednesday",
        "crystal": "aventurine",
        "herb": "mint",
    },
    6: {
        "meaning": "Harmony, love, nurturing, home, responsibility",
        "planet": "Venus",
        "element": "earth",
        "keywords": ["harmony", "love", "beauty", "nurturing", "home"],
        "tarot": "The Lovers",
        "magical_use": "Love spells, harmonizing spaces, family magick",
        "day": "Friday",
        "crystal": "rose quartz",
        "herb": "rose",
    },
    7: {
        "meaning": "Wisdom, spirituality, mystery, introspection, the seven planets",
        "planet": "Neptune",
        "element": "water",
        "keywords": ["wisdom", "mystery", "spirituality", "analysis", "seeking"],
        "tarot": "The Chariot",
        "magical_use": "Divination, meditation, seeking hidden knowledge, spiritual growth",
        "day": "Monday",
        "crystal": "labradorite",
        "herb": "mugwort",
    },
    8: {
        "meaning": "Power, abundance, infinity, karmic balance, the eight sabbats",
        "planet": "Saturn",
        "element": "earth",
        "keywords": ["power", "abundance", "karma", "mastery", "cycles"],
        "tarot": "Strength",
        "magical_use": "Prosperity magick, karmic work, personal power, the Wheel of the Year",
        "day": "Saturday",
        "crystal": "pyrite",
        "herb": "basil",
    },
    9: {
        "meaning": "Completion, wisdom, humanitarianism, the knot spell number",
        "planet": "Mars",
        "element": "fire",
        "keywords": ["completion", "wisdom", "service", "culmination", "release"],
        "tarot": "The Hermit",
        "magical_use": "Completion spells, banishing, ending cycles, knot magick",
        "day": "Tuesday",
        "crystal": "garnet",
        "herb": "rosemary",
    },
    10: {
        "meaning": "Fulfillment, new cycle, wholeness, return to one",
        "planet": "Sun",
        "element": "spirit",
        "keywords": ["fulfillment", "completion", "rebirth", "wholeness"],
        "tarot": "Wheel of Fortune",
        "magical_use": "Major transitions, celebrating achievements, starting fresh",
        "day": "Sunday",
        "crystal": "clear quartz",
        "herb": "frankincense",
    },
    11: {
        "meaning": "Master number — intuition, spiritual awakening, illumination",
        "planet": "Moon",
        "element": "spirit",
        "keywords": ["intuition", "illumination", "mastery", "vision", "gateway"],
        "tarot": "Justice",
        "magical_use": "Divination, spiritual breakthrough, psychic development",
        "day": "Monday",
        "crystal": "selenite",
        "herb": "star anise",
    },
    12: {
        "meaning": "Cosmic order, completion of cycles, the zodiac",
        "planet": "Jupiter",
        "element": "spirit",
        "keywords": ["cosmic order", "zodiac", "sacrifice", "perspective"],
        "tarot": "The Hanged Man",
        "magical_use": "Surrender spells, gaining new perspective, yearly rituals",
        "day": "Thursday",
        "crystal": "fluorite",
        "herb": "cedar",
    },
    13: {
        "meaning": "Transformation, death and rebirth, the 13 lunar months",
        "planet": "Moon",
        "element": "water",
        "keywords": ["transformation", "rebirth", "lunar", "mystery", "the Goddess"],
        "tarot": "Death",
        "magical_use": "Deep transformation, ending and beginning, esbat work",
        "day": "Monday",
        "crystal": "obsidian",
        "herb": "myrrh",
    },
}


# ── Helper functions ───────────────────────────────────────────────────────

def get_number_meaning(number: int) -> dict | None:
    """Get correspondences for a number (1-13)."""
    return NUMBERS.get(number)


def reduce_to_single(number: int) -> int:
    """Reduce a number to a single digit (1-9) or master number (11, 13).
    Standard numerological reduction, preserving 11 and 13 as master numbers.
    """
    if number in (11, 13):
        return number
    while number > 9:
        number = sum(int(d) for d in str(number))
        if number in (11, 13):
            return number
    return number


def name_to_number(name: str) -> int:
    """Convert a name/word to its numerological value.
    Uses Pythagorean system: A=1, B=2... I=9, J=1, K=2... etc.
    """
    values = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
        'j': 1, 'k': 2, 'l': 3, 'm': 4, 'n': 5, 'o': 6, 'p': 7, 'q': 8, 'r': 9,
        's': 1, 't': 2, 'u': 3, 'v': 4, 'w': 5, 'x': 6, 'y': 7, 'z': 8,
    }
    total = sum(values.get(c, 0) for c in name.lower() if c.isalpha())
    return reduce_to_single(total)


def date_to_number(year: int, month: int, day: int) -> int:
    """Calculate the numerological value of a date."""
    total = sum(int(d) for d in f"{year}{month:02d}{day:02d}")
    return reduce_to_single(total)


def get_magical_number(intention: str) -> dict:
    """Get the best number for a magical intention."""
    intention_lower = intention.lower()
    intention_map = {
        "protection": 4, "love": 6, "prosperity": 8, "healing": 2,
        "divination": 7, "banishing": 9, "cleansing": 2, "creativity": 3,
        "wisdom": 7, "confidence": 1, "communication": 5, "grounding": 4,
        "transformation": 13, "peace": 2, "courage": 9, "new beginning": 1,
        "intuition": 11, "abundance": 8, "balance": 6, "spiritual": 7,
    }
    for keyword, num in intention_map.items():
        if keyword in intention_lower:
            return {"number": num, **NUMBERS[num]}
    # Default: calculate from the intention text
    num = name_to_number(intention)
    return {"number": num, **NUMBERS.get(num, NUMBERS[1])}
