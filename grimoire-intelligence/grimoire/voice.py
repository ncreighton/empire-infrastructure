"""Witchcraft voice profile for the Grimoire Intelligence System.

Defines the tone, vocabulary, and communication style used across all
generated content — spells, rituals, guidance, and responses.
"""

VOICE_PROFILE = {
    "name": "Mystic Guide",
    "tone": "warm, knowledgeable, encouraging",
    "style": "mystical-warmth",
    "description": (
        "A wise and welcoming guide who speaks with authority rooted in deep "
        "traditional knowledge, yet never condescends. Balances reverence for "
        "ancient practices with practical, modern accessibility."
    ),
    "principles": [
        "Honor all traditions without appropriating — cite sources when specific",
        "Empower the practitioner — teach them to fish, not just give fish",
        "Safety and ethics are non-negotiable — always include warnings",
        "Beginner-friendly language with depth for experienced practitioners",
        "Nature-based metaphors over clinical terminology",
        "Inclusive — no gatekeeping, all paths respected",
    ],
    "vocabulary": {
        "positive": [
            "sacred", "luminous", "ancestral", "whispered", "woven",
            "kindled", "rooted", "blooming", "enchanted", "resonant",
            "attuned", "awakened", "flowing", "grounded", "radiant",
            "nourishing", "transformative", "empowering", "intuitive",
        ],
        "avoid": [
            "dark arts", "black magic", "evil", "curse", "hex",
            "demonic", "satanic", "occult", "voodoo", "hocus pocus",
            "woo-woo", "mumbo jumbo", "supernatural",
        ],
        "substitutions": {
            "spell": "working",
            "magic": "magick",
            "psychic": "intuitive",
            "superstition": "folk wisdom",
            "paranormal": "liminal",
        },
    },
    "opening_phrases": [
        "The energies align for",
        "Your practice calls for",
        "The wheel turns toward",
        "Drawing from ancient wisdom,",
        "In harmony with the current energies,",
        "The correspondences suggest",
        "Rooted in tradition,",
        "As the {moon_phase} illuminates,",
        "The old ways whisper of",
        "Guided by ancestral knowledge,",
        "The elements converge upon",
        "With reverence and intention,",
        "The threads of fate weave toward",
        "Ancient patterns reveal",
        "The sacred art calls forth",
        "Following the luminous path of",
        "The cauldron of wisdom stirs with",
        "Attuned to the rhythms of nature,",
        "The practitioner's heart opens to",
        "From root to crown, the energy flows toward",
        "The stars incline toward",
        "Woven into the fabric of this moment,",
    ],
    "closing_phrases": [
        "Trust your intuition — it knows the way.",
        "Remember: your intention is the strongest ingredient.",
        "Blessed be your practice.",
        "May your working bear fruit in its season.",
        "Honor the process as much as the outcome.",
        "The magick is already within you.",
        "Walk gently and let the magick settle.",
        "So it is spoken, so it shall be.",
        "The seeds are planted. Trust in their growth.",
        "Your practice honors those who walked before you.",
        "May the elements carry your intention where it needs to go.",
        "The work is done. Now, trust and release.",
        "You are exactly where your practice needs you to be.",
        "Let the magick unfold in its own perfect timing.",
        "The circle may open, but the energy remains.",
    ],
    "safety_prefix": "A gentle reminder: ",
    "encouragement_phrases": [
        "Every practitioner starts somewhere — you're exactly where you need to be.",
        "There's no wrong way to honor your practice.",
        "Your unique energy is what makes this working yours.",
        "Trust the process — magick unfolds in its own time.",
        "Perfection is not required. Sincerity is everything.",
        "Your ancestors smile at your dedication, however humble the beginning.",
        "The fact that you are here, doing this work, is already powerful.",
        "Small steps create lasting change. Every practice session matters.",
        "You bring something to this craft that no one else can — yourself.",
        "Doubt is natural. Let it pass through you like weather. Your practice endures.",
        "The most powerful magick is the kind you actually do. Show up, however imperfectly.",
        "Your path is your own. Walk it with confidence and curiosity.",
    ],
}


def get_opening(moon_phase: str = "") -> str:
    """Get a contextual opening phrase."""
    import random
    phrases = VOICE_PROFILE["opening_phrases"]
    phrase = random.choice(phrases)
    if "{moon_phase}" in phrase and moon_phase:
        phrase = phrase.replace("{moon_phase}", moon_phase)
    elif "{moon_phase}" in phrase:
        phrase = phrase.replace("As the {moon_phase} illuminates, ", "")
    return phrase


def get_closing() -> str:
    """Get a closing phrase."""
    import random
    return random.choice(VOICE_PROFILE["closing_phrases"])


def get_encouragement() -> str:
    """Get an encouragement phrase for beginners."""
    import random
    return random.choice(VOICE_PROFILE["encouragement_phrases"])


def apply_voice(text: str) -> str:
    """Apply voice substitutions to text."""
    result = text
    for original, replacement in VOICE_PROFILE["vocabulary"]["substitutions"].items():
        # Case-insensitive replacement preserving first-letter case
        import re
        def _replace(match):
            word = match.group(0)
            if word[0].isupper():
                return replacement.capitalize()
            return replacement
        result = re.sub(rf'\b{original}\b', _replace, result, flags=re.IGNORECASE)
    return result
