"""Spell type frameworks and templates for SpellSmith generation."""

SPELL_TYPES = {
    "candle": {
        "name": "Candle Spell",
        "description": "One of the most accessible forms of magick, using flame as a focal point for intention.",
        "difficulty": "beginner",
        "duration_minutes": 20,
        "core_materials": ["candle (color matched to intention)", "matches or lighter", "fireproof holder"],
        "optional_materials": ["carving tool", "anointing oil", "herbs for dressing"],
        "structure": [
            "Cleanse your space and center yourself",
            "Carve your intention or sigil into the candle",
            "Anoint the candle with oil (tip to middle for drawing in, middle to tip for releasing)",
            "Roll candle in dried herbs if desired",
            "Place candle in holder and light it",
            "State your intention clearly, aloud or in your mind",
            "Meditate on the flame, visualizing your intention manifesting",
            "Allow the candle to burn completely if safe, or snuff (never blow) and relight daily",
        ],
        "closing": "Thank the element of fire and any deities or spirits invoked. Ground yourself.",
        "safety": [
            "Never leave a burning candle unattended",
            "Keep away from flammable materials",
            "Use a fireproof surface and holder",
            "Snuff candles rather than blowing them out",
        ],
        "tips": [
            "Color matters: match candle color to your intention",
            "Carve symbols or words that represent your goal",
            "Dressing a candle with oil amplifies the working",
            "Birthday candles work great for quick spells",
        ],
        "best_for": ["manifestation", "protection", "love", "prosperity", "healing"],
    },
    "jar": {
        "name": "Jar Spell (Witch Bottle)",
        "description": "A container spell that holds and amplifies your intention over time.",
        "difficulty": "beginner",
        "duration_minutes": 30,
        "core_materials": ["glass jar with lid", "paper and pen", "herbs and/or crystals"],
        "optional_materials": ["honey", "salt", "personal concerns", "wax to seal"],
        "structure": [
            "Cleanse your jar with smoke, sound, or moonlight",
            "Write your intention on the paper",
            "Layer ingredients into the jar with intention",
            "Add herbs, crystals, and any liquids",
            "Fold the paper toward you (to draw in) or away (to banish) and add to jar",
            "Seal the lid and drip candle wax over it if desired",
            "Shake the jar while stating your intention",
            "Place the jar where it can do its work",
        ],
        "closing": "The jar continues working as long as it remains sealed. Revisit and shake periodically to reactivate.",
        "safety": [
            "Some ingredients may react — research before combining",
            "If using honey, ensure lid is tight to avoid pests",
            "Dispose respectfully when done: bury for earth magick, pour at crossroads for release",
        ],
        "tips": [
            "Honey jars sweeten situations and relationships",
            "Salt jars protect and ward",
            "Vinegar jars sour unwanted situations",
            "Shake your jar to reactivate the spell",
        ],
        "best_for": ["long-term goals", "protection", "love sweetening", "prosperity", "banishing"],
    },
    "knot": {
        "name": "Knot Spell (Cord Magick)",
        "description": "Binding intention into physical knots, an ancient and portable form of magick.",
        "difficulty": "beginner",
        "duration_minutes": 15,
        "core_materials": ["cord, ribbon, or string (color matched)", "scissors"],
        "optional_materials": ["essential oils", "herbs to tie in", "beads"],
        "structure": [
            "Choose a cord in a color aligned with your intention",
            "Cleanse the cord by passing it through incense smoke or moonlight",
            "Hold the cord and focus on your intention",
            "Tie nine knots in this traditional order: 1-6-4-7-3-8-5-9-2",
            "With each knot, speak a line of your intention or chant",
            "Traditional chant: 'By knot of one, the spell's begun...'",
            "When all nine knots are tied, the spell is sealed",
            "Keep the cord safe, or release by untying/burning when goal is met",
        ],
        "closing": "Store the knotted cord in a safe place. To release the energy, untie the knots one by one or burn the cord.",
        "safety": [
            "Be mindful of fire safety if burning cords",
            "Natural fibers (cotton, silk, wool) are preferred over synthetic",
        ],
        "tips": [
            "Nine is the traditional number but three or seven also work",
            "Braid three cords together for extra power",
            "Untie one knot each day to release energy gradually",
            "Great for travel — portable and discreet",
        ],
        "best_for": ["binding", "commitment", "patience", "long-term goals", "protection"],
    },
    "sachet": {
        "name": "Sachet (Mojo Bag / Charm Bag)",
        "description": "A portable bundle of magical ingredients carried on your person.",
        "difficulty": "beginner",
        "duration_minutes": 20,
        "core_materials": ["small cloth bag or fabric square with ribbon", "herbs", "crystal chips"],
        "optional_materials": ["petition paper", "personal item", "essential oil drops"],
        "structure": [
            "Choose fabric color aligned with your intention",
            "Lay out your chosen herbs, crystals, and items",
            "Hold each ingredient and speak its purpose aloud",
            "Place items into the bag one at a time with intention",
            "Add a written petition or sigil if desired",
            "Tie the bag closed with three knots",
            "Breathe your intention into the bag",
            "Carry it on your person or place it where needed",
        ],
        "closing": "Feed your sachet periodically with essential oil, breath, or moonlight to keep it active.",
        "safety": [
            "Ensure herbs are safe for skin contact if carrying against body",
            "Replace fresh herbs every moon cycle",
        ],
        "tips": [
            "Traditional number of ingredients is odd: 3, 5, 7, or 9",
            "Feed your mojo bag weekly to keep it charged",
            "Don't let others touch your sachet",
            "Refresh at each full moon for ongoing workings",
        ],
        "best_for": ["daily protection", "luck", "love attraction", "confidence", "travel safety"],
    },
    "crystal_grid": {
        "name": "Crystal Grid",
        "description": "A geometric arrangement of crystals that amplifies and focuses intention.",
        "difficulty": "intermediate",
        "duration_minutes": 45,
        "core_materials": ["center stone (focus crystal)", "6-12 supporting crystals", "grid cloth or template"],
        "optional_materials": ["crystal wand for activation", "printed sacred geometry template", "candles"],
        "structure": [
            "Cleanse all crystals and your grid space",
            "Set your intention clearly — write it down",
            "Place your center stone first, stating your intention",
            "Working outward, place crystals in your chosen geometric pattern",
            "Common patterns: Flower of Life, hexagonal, spiral",
            "Activate the grid by pointing a wand or finger from outer to center",
            "Visualize lines of energy connecting each crystal",
            "Leave the grid undisturbed for the duration of your working",
        ],
        "closing": "Dismantle the grid in reverse order when your intention manifests or after one moon cycle.",
        "safety": [
            "Some crystals are water-soluble — don't cleanse with water",
            "Place grid where it won't be disturbed by pets or children",
        ],
        "tips": [
            "Clear quartz points amplify any grid",
            "Photograph your grid to preserve the energy pattern",
            "Reactivate weekly by tracing the energy lines",
            "Sacred geometry templates are available online",
        ],
        "best_for": ["amplification", "healing", "manifestation", "home protection", "meditation focus"],
    },
    "mirror": {
        "name": "Mirror Spell",
        "description": "Uses reflective surfaces for scrying, protection, or returning negative energy.",
        "difficulty": "intermediate",
        "duration_minutes": 30,
        "core_materials": ["small mirror", "candle", "black cloth for covering"],
        "optional_materials": ["protective herbs", "sigils", "moon water"],
        "structure": [
            "Cleanse the mirror with moon water or smoke",
            "Set up in a dim room with candle light",
            "State your intention clearly to the mirror",
            "For protection: face mirror outward to reflect negativity",
            "For scrying: gaze softly at the mirror surface, letting vision blur",
            "For self-work: speak affirmations to your reflection",
            "Hold focus for at least 10 minutes",
            "Cover the mirror with black cloth when finished",
        ],
        "closing": "Always cover or face down mirrors used in magick when not in active use.",
        "safety": [
            "Grounding is essential after mirror work",
            "Cover mirrors when not in use for magical purposes",
            "Not recommended for those experiencing high anxiety",
            "Always have a grounding object nearby",
        ],
        "tips": [
            "Black mirrors (obsidian) are traditional for scrying",
            "Start with short sessions and build up tolerance",
            "Keep a journal of what you see during scrying",
            "Full moon is ideal for mirror consecration",
        ],
        "best_for": ["protection", "divination", "shadow work", "self-reflection", "returning negativity"],
    },
    "bath": {
        "name": "Ritual Bath",
        "description": "Water magick that cleanses, charges, and transforms through full immersion.",
        "difficulty": "beginner",
        "duration_minutes": 30,
        "core_materials": ["bathtub or large basin", "salt (sea or Epsom)", "herbs or essential oils"],
        "optional_materials": ["candles", "crystals (water-safe only)", "flower petals", "music"],
        "structure": [
            "Clean the bathroom physically before beginning",
            "Set up candles and any crystals around the tub",
            "Run the bath at a comfortable temperature",
            "Add salt first (for cleansing), then herbs or oils",
            "Stir the water clockwise (to draw in) or counterclockwise (to release)",
            "Enter the bath mindfully, feeling the water embrace you",
            "Soak for at least 20 minutes, visualizing your intention",
            "When draining, visualize what you're releasing going down the drain",
        ],
        "closing": "Pat dry gently — don't rub away the magical residue. Ground yourself with a warm drink.",
        "safety": [
            "Test water temperature before entering",
            "Research herb safety for skin contact and allergies",
            "Some crystals are toxic in water — research first",
            "Stay hydrated — drink water before and after",
            "Water-safe crystals: quartz, amethyst, rose quartz. AVOID: selenite, malachite, pyrite",
        ],
        "tips": [
            "A shower works too — place herbs in a muslin bag under the spray",
            "Moon water adds extra lunar energy",
            "Foot baths are equally effective if no bathtub is available",
            "Match salt type to intention: sea salt for ocean energy, Epsom for muscle tension",
        ],
        "best_for": ["cleansing", "self-love", "healing", "stress relief", "new beginnings"],
    },
    "sigil": {
        "name": "Sigil Magick",
        "description": "Creating and charging a unique magical symbol that encodes your intention.",
        "difficulty": "beginner",
        "duration_minutes": 25,
        "core_materials": ["paper", "pen or marker"],
        "optional_materials": ["colored ink", "candle for burning", "anointing oil"],
        "structure": [
            "Write your intention as a clear, present-tense statement",
            "Remove all vowels and duplicate consonants",
            "Arrange remaining letters into an abstract symbol",
            "Refine the design until it feels right — aesthetics matter",
            "Enter a focused, meditative state",
            "Charge the sigil: gaze at it intensely, chant, or use elemental charging",
            "Activate by fire (burning), water (dissolving), earth (burying), or air (releasing)",
            "Release attachment to the outcome — forget the sigil",
        ],
        "closing": "The key to sigil magick is forgetting. Once activated, release it from your conscious mind.",
        "safety": [
            "Burn paper safely in a fireproof container",
            "Be specific in your intention wording",
        ],
        "tips": [
            "The statement 'I have financial abundance' becomes letters: HFNCLBDW",
            "Charge during the planetary hour that matches your intention",
            "Digital sigils work too — create and delete the file",
            "Some practitioners draw sigils on their body with washable ink",
        ],
        "best_for": ["any intention", "quick magick", "discreet practice", "creativity", "manifestation"],
    },
}


# ── Ritual structure templates ─────────────────────────────────────────────

RITUAL_STRUCTURE = {
    "simple": {
        "name": "Simple Ritual",
        "difficulty": "beginner",
        "duration_minutes": 20,
        "phases": [
            {"name": "Ground & Center", "duration": 3, "description": "Breathe deeply, feel your connection to the earth"},
            {"name": "Set Intention", "duration": 2, "description": "State clearly what you wish to accomplish"},
            {"name": "Main Working", "duration": 10, "description": "Perform your spell, meditation, or offering"},
            {"name": "Close & Ground", "duration": 5, "description": "Thank energies invoked, ground excess energy"},
        ],
    },
    "circle": {
        "name": "Full Circle Ritual",
        "difficulty": "intermediate",
        "duration_minutes": 45,
        "phases": [
            {"name": "Purify Space", "duration": 5, "description": "Smoke cleanse, sweep, or sprinkle salt water"},
            {"name": "Cast Circle", "duration": 5, "description": "Walk the boundary clockwise, visualize protective barrier"},
            {"name": "Call Quarters", "duration": 5, "description": "Invoke the four elements at their cardinal directions"},
            {"name": "Invoke Deity/Spirit", "duration": 3, "description": "Call upon any deities or spirits for your working"},
            {"name": "State Intention", "duration": 2, "description": "Declare the purpose of your ritual"},
            {"name": "Main Working", "duration": 15, "description": "Core spell, meditation, or ceremony"},
            {"name": "Cakes & Ale", "duration": 3, "description": "Share food and drink to ground and celebrate"},
            {"name": "Thank & Release", "duration": 3, "description": "Thank spirits, release quarters in reverse order"},
            {"name": "Open Circle", "duration": 2, "description": "Walk counterclockwise, declare circle open"},
            {"name": "Ground", "duration": 2, "description": "Eat, drink, touch the earth"},
        ],
    },
    "sabbat": {
        "name": "Sabbat Celebration",
        "difficulty": "intermediate",
        "duration_minutes": 60,
        "phases": [
            {"name": "Prepare Sacred Space", "duration": 10, "description": "Decorate altar with seasonal items, cleanse space"},
            {"name": "Cast Circle", "duration": 5, "description": "Create sacred container for the celebration"},
            {"name": "Call Quarters", "duration": 5, "description": "Invoke elements with seasonal associations"},
            {"name": "Seasonal Invocation", "duration": 5, "description": "Honor the turning of the wheel"},
            {"name": "Mythic Narration", "duration": 5, "description": "Tell the story of the sabbat"},
            {"name": "Main Celebration", "duration": 15, "description": "Seasonal activities, crafts, or spell work"},
            {"name": "Feast", "duration": 10, "description": "Share seasonal foods and drinks"},
            {"name": "Divination", "duration": 5, "description": "Draw cards or scry for the coming season"},
            {"name": "Close", "duration": 5, "description": "Thank, release, and open circle"},
        ],
    },
}


# ── Helper functions ───────────────────────────────────────────────────────

def get_spell_template(spell_type: str) -> dict | None:
    """Get template for a spell type."""
    return SPELL_TYPES.get(spell_type.lower())


def get_ritual_structure(structure_type: str) -> dict | None:
    """Get a ritual structure template."""
    return RITUAL_STRUCTURE.get(structure_type.lower())


def get_all_spell_types() -> list[str]:
    """Return all available spell type keys."""
    return list(SPELL_TYPES.keys())


def get_spell_types_for_intention(intention: str) -> list[str]:
    """Return spell types that work well for a given intention."""
    intention_lower = intention.lower()
    matches = []
    for key, template in SPELL_TYPES.items():
        for purpose in template["best_for"]:
            if purpose in intention_lower or intention_lower in purpose:
                matches.append(key)
                break
    return matches if matches else ["candle", "sigil"]  # defaults
